# Công thức của từng bước, giải thích từng ký hiệu

Ghi chú này viết lại mỗi bước của hệ thống thành công thức. Sau mỗi công thức là một bảng nói rõ từng ký hiệu nghĩa là
gì, nhận giá trị nào trong dự án, kèm một ví dụ bằng số. Thứ tự các bước giống trang
[`project_explained_vi.html`](project_explained_vi.html) và [`how_it_works_simple_vi.md`](how_it_works_simple_vi.md).
Mỗi công thức đều được đối chiếu với mã nguồn trong `neural_painter/`.

> Báo cáo (`report/chapters/02_theory.md`) viết dãy tham số của một nét là $\boldsymbol{\theta}$. Ghi chú này viết là
> $\mathbf{p}$, để khỏi nhầm với góc xoay $\theta$ của nét.

## 0. Ký hiệu dùng chung

| Ký hiệu | Nghĩa | Giá trị trong dự án |
| :--- | :--- | :--- |
| $m, n$ | số hàng, số cột của ảnh gốc | ví dụ $3000, 4000$ |
| $i, j$ | chỉ số hàng, cột của ảnh gốc | $i = 0 \dots m-1$, $j = 0 \dots n-1$ |
| $c$ | kênh màu | $R$, $G$ hoặc $B$ |
| $S$ | cạnh ảnh sau khi thu nhỏ | $512$ |
| $s$ | cạnh một ô (khối) mà mạng neural vẽ được | $32$ |
| $M$ | lưới mịn nhất | $5$ |
| $g$ | lưới đang vẽ, gồm $g \times g$ ô | $1, 2, \dots, 5$ |
| $k$ | chỉ số ô trong lưới | $0 \dots g^2 - 1$ |
| $N$ | số nét người dùng chọn | $500$ |
| $N_b$ | số nét mỗi ô | $9$ |
| $\ell$ | chỉ số nét trong một ô | $1 \dots N_b$ |
| $d$ | số tham số của một nét | $12$ với cọ sơn dầu |
| $\mathbf{p}$ | dãy tham số của một nét | $\mathbf{p} \in [0, 1]^{12}$ |
| $T$ | ảnh đích (target): ảnh mà tranh phải giống | |
| $C$ | canvas: bức tranh đang vẽ | |
| $\odot$ | nhân từng phần tử (pixel với pixel) | |
| $\lfloor x \rfloor$ | làm tròn xuống (phần nguyên) | $\lfloor 5.86 \rfloor = 5$ |
| $\operatorname{clip}(x, a, b)$ | kẹp $x$ vào khoảng $[a, b]$: $\min(b, \max(a, x))$ | $\operatorname{clip}(1.2, 0, 1) = 1$ |

---

## 1. Ảnh là một bảng số

Mã: `core/image_io.py`, hàm `load_image`.

$$
A \in \{0, 1, \dots, 255\}^{m \times n \times 3}, \qquad A[i, j, c] = \text{mức của kênh } c \text{ tại pixel hàng } i \text{, cột } j.
$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $A$ | ảnh vừa đọc từ file |
| $\{0, 1, \dots, 255\}$ | mỗi số là một số nguyên từ 0 đến 255 |
| $m \times n \times 3$ | bảng có $m$ hàng, $n$ cột, mỗi ô 3 số (R, G, B) |
| $A[i, j, c]$ | một con số: kênh $c$ của pixel ở hàng $i$, cột $j$ |

**Ví dụ:** pixel cam nâu ở hàng 6, cột 7 có $A[6, 7, R] = 200$, $A[6, 7, G] = 120$, $A[6, 7, B] = 40$. Ảnh
$3000 \times 4000$ có $3000 \cdot 4000 \cdot 3 = 36$ triệu con số.

---

## 2. Chuẩn hoá về $[0, 1]$

Mã: `image_io.py`, dòng 61 đến 67.

$$
\tilde{A}[i, j, c] = \frac{A[i, j, c]}{V_{\max}}, \qquad V_{\max} = 2^{\text{số bit}} - 1 =
\begin{cases} 255 & \text{ảnh 8 bit (JPG, PNG thường)} \\ 65535 & \text{ảnh xám 16 bit (PNG, TIFF)} \end{cases}
$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $\tilde{A}$ | ảnh đã chuẩn hoá, mỗi số là số thực trong $[0, 1]$ |
| $V_{\max}$ | giá trị lớn nhất mà định dạng ảnh lưu được |
| số bit | số bit dùng để lưu một kênh: 8 hoặc 16 |

**Ví dụ:** $(200, 120, 40) / 255 = (0.784, 0.471, 0.157)$. 0 vẫn là đen, 255 thành 1 (trắng).

**Chiều ngược lại**, khi lưu tranh cuối $P$ thành file PNG / JPG (`image_io.py`, hàm `to_uint8`):

$$
P_{8}[i, j, c] = \big\lfloor 255 \cdot \operatorname{clip}(P[i, j, c], 0, 1) + 0.5 \big\rfloor .
$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $P$ | tranh cuối, số thực trong $[0, 1]$ |
| $P_8$ | tranh dạng số nguyên 0 đến 255 để ghi ra file |
| $\operatorname{clip}(\cdot, 0, 1)$ | phòng khi có số lọt ra ngoài $[0, 1]$ |
| $+0.5$ rồi $\lfloor \cdot \rfloor$ | làm tròn tới số nguyên gần nhất: $\lfloor 199.6 + 0.5 \rfloor = 200$ |

---

## 3. Thu nhỏ về $S \times S$

Mã: `image_io.py`, hàm `fit` và `resize` (`cv2.INTER_AREA`). Chế độ mặc định là *stretch* (ép thẳng về hình vuông).

$$
s_y = \frac{m}{S}, \qquad s_x = \frac{n}{S},
$$

$$
I[u, v, c] = \frac{1}{s_y \, s_x} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} a_{u,i} \; b_{v,j} \; \tilde{A}[i, j, c].
$$

| Ký hiệu | Nghĩa | Ví dụ ($3000 \times 4000 \rightarrow 512 \times 512$) |
| :--- | :--- | :--- |
| $I$ | ảnh sau khi thu nhỏ, cỡ $S \times S \times 3$ | $512 \times 512 \times 3$ |
| $u, v$ | hàng, cột của ảnh mới | $0 \dots 511$ |
| $s_y$ | số hàng cũ gộp thành 1 hàng mới | $3000 / 512 \approx 5.86$ |
| $s_x$ | số cột cũ gộp thành 1 cột mới | $4000 / 512 \approx 7.81$ |
| $a_{u,i}$ | phần của hàng cũ $i$ nằm trong hàng mới $u$, từ 0 đến 1 | xem bên dưới |
| $b_{v,j}$ | phần của cột cũ $j$ nằm trong cột mới $v$, từ 0 đến 1 | |
| $\frac{1}{s_y s_x}$ | chia cho tổng trọng số, để tổng có trọng số thành **trung bình** | $\frac{1}{5.86 \cdot 7.81} \approx \frac{1}{46}$ |

Hàng mới $u$ phủ đoạn $[u \, s_y, (u + 1) \, s_y)$ trên trục hàng cũ, và $a_{u,i}$ là độ dài phần chung của đoạn đó với
hàng cũ $[i, i + 1)$. Ví dụ hàng mới đầu tiên ($u = 0$) phủ $[0, 5.86)$:

$$
a_{0,0} = a_{0,1} = a_{0,2} = a_{0,3} = a_{0,4} = 1, \qquad a_{0,5} = 0.86, \qquad a_{0,i} = 0 \text{ với } i \ge 6 .
$$

Tổng các $a_{u,i}$ của một hàng mới bằng $s_y$, tổng các $b_{v,j}$ bằng $s_x$. Vì vậy mỗi pixel mới là **trung bình cộng có
trọng số theo diện tích** của khoảng 46 pixel cũ.

**Hai chế độ khác:**

- *Crop* (khi $n > m$): giữ các cột từ khoảng $\frac{n - m}{2}$ đến $\frac{n + m}{2}$, được ảnh $m \times m$, rồi thu nhỏ
  với $s_y = s_x = m / S$. Ảnh không bị méo nhưng mất hai mép.
- *Pad* (khi $n > m$): thêm khoảng $\frac{n - m}{2}$ hàng giá trị $1$ (trắng) ở trên và dưới, được ảnh $n \times n$, rồi
  thu nhỏ với $s_y = s_x = n / S$.

**Kích thước tranh trả về** (`image_io.py`, hàm `output_size`, dùng ở chế độ *stretch*):

$$
(H_{\text{ra}}, W_{\text{ra}}) =
\begin{cases}
\left( \left\lfloor S \cdot \dfrac{m}{n} \right\rfloor ,\; S \right) & \text{nếu } m < n \text{ (ảnh ngang)} \\[2ex]
\left( S ,\; \left\lfloor S \cdot \dfrac{n}{m} \right\rfloor \right) & \text{nếu } m \ge n \text{ (ảnh dọc hoặc vuông)}
\end{cases}
$$

| Ký hiệu | Nghĩa | Ví dụ |
| :--- | :--- | :--- |
| $H_{\text{ra}}, W_{\text{ra}}$ | chiều cao, chiều rộng của tranh trả về | $\lfloor 512 \cdot 3000 / 4000 \rfloor = 384$, nên $384 \times 512$ |

---

## 4. Lưới, số nét mỗi ô, đổi toạ độ

Mã: `core/grid.py` (`grid_levels`, `strokes_per_block`, `img2patches`, `block_to_global`).

**Số nét mỗi ô:**

$$
N_b = \left\lfloor \frac{N}{\sum_{g=1}^{M} g^2} \right\rfloor, \qquad
\sum_{g=1}^{M} g^2 = \frac{M (M + 1)(2M + 1)}{6} .
$$

| Ký hiệu | Nghĩa | Ví dụ |
| :--- | :--- | :--- |
| $\sum_{g=1}^{M} g^2$ | tổng số ô của tất cả các vòng | $1 + 4 + 9 + 16 + 25 = 55$ |
| $N_b$ | số nét mỗi ô | $\lfloor 500 / 55 \rfloor = 9$ |
| $N_b \cdot \sum g^2$ | số nét thật sự được vẽ | $9 \cdot 55 = 495$ (5 nét dư không vẽ) |

**Cắt ô.** Ở vòng $g$, ảnh $I$ được thu nhỏ (lại bằng trung bình) thành $I_g$ cỡ $g s \times g s$, rồi cắt thành $g^2$
ô, đánh số theo hàng:

$$
T_g[k, c, y, x] = I_g\big[\, s \, r_k + y ,\; s \, q_k + x ,\; c \,\big], \qquad
r_k = \left\lfloor \frac{k}{g} \right\rfloor, \quad q_k = k \bmod g .
$$

| Ký hiệu | Nghĩa | Ví dụ ($g = 5$, $k = 13$) |
| :--- | :--- | :--- |
| $I_g$ | ảnh thu về cỡ $g s \times g s$ | $160 \times 160$ |
| $T_g$ | tất cả các ô của vòng $g$, cỡ $g^2 \times 3 \times s \times s$ | $25 \times 3 \times 32 \times 32$ |
| $r_k$ | hàng của ô $k$ trong lưới | $\lfloor 13 / 5 \rfloor = 2$ |
| $q_k$ | cột của ô $k$ trong lưới ($\bmod$ là phép lấy dư) | $13 \bmod 5 = 3$ |
| $y, x$ | hàng, cột của pixel bên trong ô | $0 \dots 31$ |

**Đổi toạ độ nét từ ô ra cả ảnh:**

$$
x_c^{\text{ảnh}} = \frac{q_k + x_c^{\text{ô}}}{g}, \qquad
y_c^{\text{ảnh}} = \frac{r_k + y_c^{\text{ô}}}{g}, \qquad
w^{\text{ảnh}} = \frac{w^{\text{ô}}}{g}, \qquad
h^{\text{ảnh}} = \frac{h^{\text{ô}}}{g} .
$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $x_c^{\text{ô}}, y_c^{\text{ô}}$ | tâm nét tính trong ô, từ 0 đến 1 |
| $x_c^{\text{ảnh}}, y_c^{\text{ảnh}}$ | tâm nét tính trên cả ảnh, từ 0 đến 1 |
| $w, h$ | chiều rộng, chiều cao của nét; ô nhỏ hơn $g$ lần nên nét cũng nhỏ hơn $g$ lần |

**Ví dụ:** ô $k = 13$ của lưới $5 \times 5$, nét ở giữa ô ($x_c^{\text{ô}} = y_c^{\text{ô}} = 0.5$):
$x_c^{\text{ảnh}} = (3 + 0.5) / 5 = 0.7$ và $y_c^{\text{ảnh}} = (2 + 0.5) / 5 = 0.5$, tức pixel khoảng $(358, 256)$ trên ảnh 512.

---

## 5. Nét cọ và cách ghép nét

Mã: `configs/oil_brush.yaml`, `core/procedural_rasterizer.py`, `core/differentiable_canvas.py`.

**Dãy tham số của một nét sơn dầu:**

$$
\mathbf{p} = \big( x_c,\; y_c,\; w,\; h,\; \theta,\; \mathbf{c}_0,\; \mathbf{c}_2,\; \alpha \big) \in [0, 1]^{12},
\qquad \mathbf{c}_0 = (r_0, g_0, b_0), \quad \mathbf{c}_2 = (r_2, g_2, b_2).
$$

| Ký hiệu | Nghĩa | Số lượng |
| :--- | :--- | :---: |
| $x_c, y_c$ | tâm nét (ngang, dọc) | 2 |
| $w, h$ | chiều rộng, chiều cao | 2 |
| $\theta$ | góc xoay, góc thật là $\pi \theta$ radian ($\theta \cdot 180°$) | 1 |
| $\mathbf{c}_0$ | màu đầu nét (đỏ, lục, lam) | 3 |
| $\mathbf{c}_2$ | màu cuối nét | 3 |
| $\alpha$ | độ che; với sơn dầu luôn bị đặt là 1 | 1 |

**Đổi sang pixel** trên một canvas cạnh $L$ pixel ($L = 512$ khi vẽ tranh cuối, $L = 32$ trong một ô):

$$
x_0 = \big\lfloor x_c (L - 1) + 0.5 \big\rfloor, \quad
y_0 = \big\lfloor y_c (L - 1) + 0.5 \big\rfloor, \quad
w_{px} = \lfloor 1 + w L \rfloor, \quad
h_{px} = \lfloor 1 + h L \rfloor .
$$

| Ký hiệu | Nghĩa | Ví dụ ($L = 512$) |
| :--- | :--- | :--- |
| $x_0, y_0$ | tâm nét tính bằng pixel | $x_c = 0.5 \Rightarrow x_0 = \lfloor 255.5 + 0.5 \rfloor = 256$ |
| $w_{px}, h_{px}$ | kích thước nét tính bằng pixel | $w = 0.3 \Rightarrow w_{px} = \lfloor 154.6 \rfloor = 154$ |

**Tô màu ảnh mẫu vệt cọ.** Ảnh mẫu $\tau$ là một ảnh xám cỡ $H_\tau \times W_\tau$, hàng $a$, cột $b$:

$$
\lambda_a = \frac{a}{H_\tau}, \qquad
\mathbf{col}(a) = (1 - \lambda_a)\, \mathbf{c}_0 + \lambda_a \, \mathbf{c}_2,
$$

$$
F^{\tau}[a, b] = \tau[a, b] \cdot \mathbf{col}(a), \qquad
A^{\tau}[a, b] = \begin{cases} 1 & \text{nếu } \tau[a, b] > 0 \\ 0 & \text{nếu không} \end{cases}
$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $\tau[a, b]$ | độ sáng của ảnh mẫu vệt cọ, từ 0 (nền đen) đến 1 |
| $\lambda_a$ | vị trí của hàng $a$ dọc theo nét: 0 ở đầu nét, gần 1 ở cuối nét |
| $\mathbf{col}(a)$ | màu của hàng $a$: pha giữa màu đầu và màu cuối |
| $F^{\tau}$ | ảnh màu của nét (vân cọ nhân với màu) |
| $A^{\tau}$ | mặt nạ độ che: 1 ở chỗ có vân cọ, 0 ở nền |

**Ví dụ:** ở giữa nét $\lambda_a = 0.5$, nên màu là trung bình của màu đầu và màu cuối.

**Kéo giãn, xoay, đặt vào vị trí.** Điểm $(b, a)$ của ảnh mẫu được đưa tới điểm $(x, y)$ trên canvas:

$$
\begin{pmatrix} x \\ y \end{pmatrix} =
\underbrace{\begin{pmatrix} \cos \pi\theta & -\sin \pi\theta \\ \sin \pi\theta & \cos \pi\theta \end{pmatrix}}_{\text{xoay}}
\underbrace{\begin{pmatrix} \frac{w_{px}}{W_\tau} \left( b - \frac{W_\tau}{2} \right) \\[1ex] \frac{h_{px}}{H_\tau} \left( a - \frac{H_\tau}{2} \right) \end{pmatrix}}_{\text{đưa tâm về gốc rồi kéo giãn}}
+ \underbrace{\begin{pmatrix} x_0 \\ y_0 \end{pmatrix}}_{\text{dời tới tâm nét}} .
$$

| Phần | Nghĩa |
| :--- | :--- |
| $b - \frac{W_\tau}{2}$, $a - \frac{H_\tau}{2}$ | dời ảnh mẫu sao cho tâm của nó nằm ở gốc toạ độ |
| $\frac{w_{px}}{W_\tau}$, $\frac{h_{px}}{H_\tau}$ | hệ số kéo giãn để ảnh mẫu thành cỡ $w_{px} \times h_{px}$ |
| ma trận $\cos / \sin$ | ma trận xoay một góc $\pi \theta$ |
| $+ (x_0, y_0)$ | dời tới tâm nét |

Kết quả là ảnh màu $F_\ell$ và mặt nạ $A_\ell$ của nét thứ $\ell$, cỡ $L \times L$.

**Ghép nét lên canvas**, lần lượt từng nét:

$$
C_\ell = A_\ell \odot F_\ell + (\mathbf{1} - A_\ell) \odot C_{\ell - 1}, \qquad
C_0 = \mathbf{1} \text{ (trắng: sơn dầu, màu nước)} \text{ hoặc } \mathbf{0} \text{ (đen: bút dạ, băng dính)}.
$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $C_\ell$ | canvas sau khi đã vẽ $\ell$ nét |
| $C_0$ | canvas trống lúc đầu |
| $F_\ell$ | ảnh màu của nét thứ $\ell$ |
| $A_\ell$ | mặt nạ độ che của nét thứ $\ell$, mỗi pixel từ 0 đến 1 |
| $\mathbf{1}, \mathbf{0}$ | ảnh toàn số 1, toàn số 0 |

**Ví dụ tại một pixel:** $A = 1$ thì pixel lấy màu nét; $A = 0$ thì giữ màu cũ; $A = 0.5$ thì lấy trung bình hai màu.
Vì đây là trung bình có trọng số của hai số trong $[0, 1]$, canvas luôn nằm trong $[0, 1]$.

---

## 6. Chọn chỗ đặt nét mới

Mã: `pipeline/stroke_sampler.py` (`error_map`, `box_blur`, `StrokeSampler`).

**Bản đồ sai lệch** của một ô:

$$
E[y, x] = \sum_{c \in \{R, G, B\}} \big| T[c, y, x] - C[c, y, x] \big| .
$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $T[c, y, x]$ | ảnh đích (một ô) tại kênh $c$, pixel $(y, x)$ |
| $C[c, y, x]$ | tranh đang vẽ tại cùng chỗ |
| $\lvert \cdot \rvert$ | trị tuyệt đối: sai lệch không phân biệt tranh sáng hơn hay tối hơn ảnh |
| $E[y, x]$ | sai lệch tại pixel $(y, x)$, từ 0 (đúng) đến 3 (sai hết cả ba kênh) |

**Làm mờ** bằng trung bình trên ô vuông $K \times K$:

$$
\bar{E}[y, x] = \frac{1}{K^2} \sum_{\delta_y} \sum_{\delta_x} E[y + \delta_y, x + \delta_x], \qquad K = \left\lfloor \frac{s}{8} \right\rfloor = 4 .
$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $\bar{E}$ | bản đồ sai lệch đã làm mờ |
| $K$ | cạnh ô vuông lấy trung bình: $32 / 8 = 4$ pixel |
| $\delta_y, \delta_x$ | độ lệch so với pixel giữa, chạy qua $K$ giá trị mỗi chiều (ở mép ảnh thì lấy đối xứng) |

**Xác suất chọn mỗi điểm:**

$$
w[y, x] = \bar{E}[y, x]^{\gamma}, \qquad
\Pr(y, x) = \frac{w[y, x]}{\sum_{y'} \sum_{x'} w[y', x']}, \qquad \gamma = 4 .
$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $\gamma$ | số mũ: chỗ sai gấp đôi được ưu tiên gấp $2^\gamma = 16$ lần |
| $w[y, x]$ | trọng số (chưa chuẩn hoá) của điểm $(y, x)$ |
| $\sum_{y'} \sum_{x'} w$ | tổng trọng số của mọi điểm; chia cho tổng để các xác suất cộng lại bằng 1 |
| $\Pr(y, x)$ | xác suất điểm $(y, x)$ được chọn làm tâm nét mới |

Trước khi bốc thăm, bản đồ được phóng lên lưới $512 \times 512$, nên tâm nét là bội số của $1/512$.

**Ví dụ:** nửa trái sai $0.8$, nửa phải sai $0.1$. Tỉ lệ ưu tiên là $(0.8 / 0.1)^4 = 8^4 = 4096$, nên nửa trái được chọn
$\frac{8 \cdot 0.4096}{8 \cdot 0.4096 + 8 \cdot 0.0001} \approx 99.98\%$ số lần.

**Tham số ban đầu của nét mới** (`procedural_rasterizer.py`, hàm `initial_params_at`):

$$
(x_c, y_c) = \text{điểm vừa bốc}, \qquad
\mathbf{c}_0 = \mathbf{c}_2 = T\big[\,:,\; \lfloor y_c s \rfloor,\; \lfloor x_c s \rfloor \,\big], \qquad
w, h \sim \mathcal{U}(0.2, 0.4), \qquad \theta \sim \mathcal{U}(0, 1).
$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $T[:, y, x]$ | cả 3 kênh màu của ảnh đích tại pixel $(y, x)$ |
| $\sim \mathcal{U}(a, b)$ | bốc ngẫu nhiên đều trong khoảng $[a, b]$ |
| $0.2$ đến $0.4$ | kích thước ban đầu tính theo cạnh ô (cấu hình sơn dầu từ tuần 6) |

---

## 7. Tinh chỉnh nét bằng gradient

Mã: `pipeline/painter_engine.py`, `losses/pixel_loss.py`.

**Hàm mất mát** (độ sai) của tất cả các ô trong một vòng:

$$
\mathcal{L}(\mathcal{P}) = \frac{1}{B \cdot 3 \cdot s^2} \sum_{k=0}^{B-1} \sum_{c} \sum_{y=0}^{s-1} \sum_{x=0}^{s-1}
\Big| C^{(k)}_{N_b}[c, y, x] - T^{(k)}[c, y, x] \Big| .
$$

| Ký hiệu | Nghĩa | Giá trị |
| :--- | :--- | :--- |
| $\mathcal{P}$ | tất cả tham số của tất cả các nét đang chỉnh | $B \times N_b \times 12$ số |
| $B$ | số ô xử lý cùng lúc ($g^2$) | 25 ở vòng $5 \times 5$ |
| $C^{(k)}_{N_b}$ | canvas của ô $k$ sau khi ghép đủ $N_b$ nét (bằng mạng neural, mục 8) | |
| $T^{(k)}$ | ảnh đích của ô $k$ | |
| $\frac{1}{B \cdot 3 \cdot s^2}$ | chia cho tổng số con số để lấy trung bình | $\frac{1}{25 \cdot 3 \cdot 1024}$ |
| $\mathcal{L}$ | độ sai trung bình mỗi con số (gọi là L1); 0 là giống hệt | |

Hàm mất mát đầy đủ trong mã là $\beta_1 \mathcal{L} + \beta_{OT} \mathcal{L}_{OT}$, với $\beta_1 = 1$ và $\beta_{OT} = 0$:
phần Sinkhorn (vận chuyển tối ưu) đã được thử, không có tác dụng, nên bị tắt.

**Gradient** và quy tắc hạ gradient cơ bản:

$$
g = \frac{\partial \mathcal{L}}{\partial p}, \qquad p \leftarrow p - \eta \, g .
$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $p$ | một con số bất kỳ trong $\mathcal{P}$, ví dụ $x_c$ của một nét |
| $\frac{\partial \mathcal{L}}{\partial p}$ | đạo hàm riêng: tăng $p$ thêm một chút thì $\mathcal{L}$ tăng (dương) hay giảm (âm), nhanh cỡ nào |
| $\eta$ | tốc độ học: độ dài mỗi bước |
| $p - \eta g$ | đi ngược dấu đạo hàm, tức đi về phía làm $\mathcal{L}$ giảm |

PyTorch tính $g$ cho mọi con số cùng lúc bằng quy tắc dây chuyền (lan truyền ngược):

$$
\frac{\partial \mathcal{L}}{\partial p} =
\frac{\partial \mathcal{L}}{\partial C} \cdot \frac{\partial C}{\partial (\hat{F}, \hat{A})} \cdot \frac{\partial (\hat{F}, \hat{A})}{\partial p} ,
$$

lần lượt: độ sai đổi theo tranh, tranh đổi theo nét (công thức ghép nét), và nét đổi theo tham số (mạng neural). Mắt xích
cuối chỉ tồn tại nhờ mạng neural; cọ thật không cho mắt xích này (mục 8).

**RMSprop**, cách bước thật sự được tính, cho từng con số:

$$
v_t = \rho \, v_{t-1} + (1 - \rho) \, g_t^2, \qquad
p_t = p_{t-1} - \eta_t \, \frac{g_t}{\sqrt{v_t} + \varepsilon}, \qquad v_0 = 0 .
$$

| Ký hiệu | Nghĩa | Giá trị |
| :--- | :--- | :--- |
| $t$ | số thứ tự của bước chỉnh | $1, 2, \dots$ |
| $g_t$ | đạo hàm ở bước $t$ | |
| $v_t$ | trung bình trượt của $g^2$: gradient gần đây lớn cỡ nào | |
| $\rho$ | mức "nhớ" các bước trước | $0.99$ (mặc định của PyTorch) |
| $\sqrt{v_t}$ | độ lớn điển hình của gradient; chia cho nó để số nào cũng bước cỡ $\eta_t$ | |
| $\varepsilon$ | số rất nhỏ để không chia cho 0 | $10^{-8}$ |

**Tốc độ học theo đường cosin**, bắt đầu lại mỗi khi thêm một nét mới:

$$
\eta_t = \eta_{\max} \cdot \frac{1}{2} \left( 1 + \cos \frac{\pi t}{n_{\text{it}}} \right), \qquad t = 0, 1, \dots, n_{\text{it}} - 1 .
$$

| Ký hiệu | Nghĩa | Giá trị |
| :--- | :--- | :--- |
| $\eta_{\max}$ | tốc độ học lớn nhất | $0.004$ |
| $n_{\text{it}}$ | số bước chỉnh sau mỗi nét mới | $\lfloor 500 / N_b \rfloor = \lfloor 500 / 9 \rfloor = 55$ ("Chất lượng tốt nhất"), 11 ("Nhanh") |
| $t$ | bước thứ mấy kể từ khi thêm nét mới | $0 \dots 54$ |

**Ví dụ:** $t = 0$: $\eta = 0.004$; $t = 27$: $\eta \approx 0.0021$; $t = 54$: $\eta \approx 0.000003$. Đầu đi nhanh, cuối
chỉnh tinh. Mỗi vòng lưới có $N_b \cdot n_{\text{it}} = 9 \cdot 55 = 495$ bước, 5 vòng là $2475$ bước.

**Kẹp giá trị** sau mỗi bước:

$$
x_c, y_c, w, h, \theta \leftarrow \operatorname{clip}(\cdot, 0.1, 0.9), \qquad
\mathbf{c}_0, \mathbf{c}_2, \alpha \leftarrow \operatorname{clip}(\cdot, 0, 1) .
$$

Hình dạng bị giữ trong $[0.1, 0.9]$ để nét không dính vào mép ô; màu và độ che phải nằm trong $[0, 1]$ như mọi giá trị ảnh.

**Chia nét theo độ sai** (chỉ ở chế độ "Nhanh", `progressive_painter.allocate`). Vòng $g$ được
$B_g \approx N \cdot g^2 / \sum_{g'} g'^2$ nét (làm tròn sao cho tổng đúng bằng $N$). Mỗi lần trao 1 nét cho ô

$$
k^{*} = \arg\max_{k \,:\, n_k < n_{\max}} \frac{e_k}{n_k + 1}, \qquad n_{\max} = \max\left(1, \left\lceil 2 \cdot \frac{B_g}{g^2} \right\rceil\right),
$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $e_k$ | độ sai trung bình $\lvert T - C \rvert$ của ô $k$ trên canvas lúc bắt đầu vòng |
| $n_k$ | số nét ô $k$ đã được nhận |
| $\arg\max$ | ô làm cho biểu thức lớn nhất |
| $n_{\max}$ | trần: mỗi ô nhận tối đa gấp đôi mức trung bình ($\lceil \cdot \rceil$ là làm tròn lên) |

Đây là phương pháp D'Hondt (dùng để chia ghế trong bầu cử): ô sai nhiều nhận nhiều nét, ô gần đúng (nền phẳng) nhận ít.

---

## 8. Mạng neural bắt chước cọ thật

Mã: `models/neural_renderer.py`, `pipeline/train_renderer.py`, `core/morphology.py`.

**Mạng neural** nhận dãy tham số và vẽ ra ô $s \times s$:

$$
(\hat{F}, \hat{A}) = G_\phi(\mathbf{p}) .
$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $G_\phi$ | mạng neural vẽ nét; $\phi$ là toàn bộ trọng số của mạng (học một lần, rồi giữ cố định khi vẽ) |
| $\hat{F}$ | ảnh màu của nét do mạng vẽ, cỡ $3 \times 32 \times 32$ (dấu mũ $\hat{\ }$: "do mạng đoán") |
| $\hat{A}$ | mặt nạ độ che do mạng vẽ |

**Kiến trúc FusionNet** (hai nhánh):

$$
m = G^{\text{hình}}_\phi(\mathbf{p}_{\text{hình}}), \qquad
\mathbf{c} = G^{\text{màu}}_\phi(\mathbf{p}), \qquad
\hat{F} = \mathbf{c} \odot m, \qquad
\hat{A} = \alpha \cdot m .
$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $\mathbf{p}_{\text{hình}}$ | 5 tham số hình dạng $(x_c, y_c, w, h, \theta)$ |
| $G^{\text{hình}}_\phi$ | nhánh hình dạng: chỉ nhìn hình dạng, vẽ ra mặt nạ $m$ (chỗ nào có nét) |
| $G^{\text{màu}}_\phi$ | nhánh màu: nhìn cả 12 số, vẽ ra trường màu $\mathbf{c}$ |
| $\mathbf{c} \odot m$ | màu chỉ giữ lại ở chỗ có nét |
| $\alpha \cdot m$ | độ che; với sơn dầu $\alpha = 1$ nên $\hat{A} = m$ |

**Huấn luyện mạng** (sai số bình phương, MSE):

$$
\mathcal{L}_{\text{vẽ}}(\phi) =
\mathbb{E}_{\mathbf{p} \sim \mathcal{U}(0,1)^{d}}
\Big[ \operatorname{mean}\big( \hat{F} - F \big)^2 + \operatorname{mean}\big( \hat{A} - A \big)^2 \Big] .
$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $\mathbf{p} \sim \mathcal{U}(0, 1)^d$ | dãy tham số bốc ngẫu nhiên đều, mỗi số trong $[0, 1]$ |
| $F, A$ | "đáp án": ảnh màu và mặt nạ do cọ thật (mục 5) vẽ cho cùng $\mathbf{p}$ |
| $(\hat{F} - F)^2$ | bình phương sai lệch, phạt nặng chỗ sai nhiều |
| $\operatorname{mean}$ | trung bình trên mọi pixel và kênh |
| $\mathbb{E}$ | kỳ vọng: trung bình trên rất nhiều nét ngẫu nhiên, mỗi lượt học 50 000 nét mới |

Học xong, $\phi$ được giữ cố định. Khi vẽ tranh, chỉ các tham số nét $\mathcal{P}$ thay đổi.

Pipeline và ứng dụng web vẽ bằng mạng **bản nhẹ đã huấn luyện sẵn của bài gốc** (`--renderer original-light`, mặc định).
Dự án cũng tự huấn luyện lại mạng theo đúng công thức trên (sơn dầu: 100 lượt, mỗi lượt 50 000 nét mới) để kiểm chứng; hai
mạng vẽ gần giống nhau (hình `report/figures/renderer_comparison_oil_light.png`).

**Nở và co mép nét** (dilate / erode $3 \times 3$), áp lên đầu ra của mạng trước khi ghép:

$$
\hat{F}'[c, y, x] = \max_{\lvert \delta_y \rvert \le 1,\, \lvert \delta_x \rvert \le 1} \hat{F}[c, y + \delta_y, x + \delta_x],
\qquad
\hat{A}'[c, y, x] = \min_{\lvert \delta_y \rvert \le 1,\, \lvert \delta_x \rvert \le 1} \hat{A}[c, y + \delta_y, x + \delta_x] .
$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $\max$ trên ô $3 \times 3$ | mỗi pixel lấy giá trị lớn nhất trong 9 pixel quanh nó: màu nét **nở** ra 1 pixel |
| $\min$ trên ô $3 \times 3$ | lấy giá trị nhỏ nhất: mặt nạ **co** lại 1 pixel |

Nhờ vậy, mọi pixel còn được che đều đọc được màu nét đầy đủ, mép nét không bị viền tối. Mẹo này đáng giá $+2.71$ dB.

---

## 9. Đo chất lượng

Mã: `pipeline/metrics.py`, `scripts/score_lpips.py`.

**PSNR:**

$$
\text{MSE} = \frac{1}{3 S^2} \sum_{c} \sum_{u} \sum_{v} \big( P[u, v, c] - I[u, v, c] \big)^2, \qquad
\text{PSNR} = 10 \log_{10} \frac{1}{\text{MSE}} \;\; (\text{dB}).
$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $P$ | tranh cuối cỡ $S \times S$ |
| $I$ | ảnh đã thu nhỏ (mục 3) mà tranh phải giống |
| $\text{MSE}$ | trung bình bình phương sai lệch trên mọi con số |
| $1$ trong $\frac{1}{\text{MSE}}$ | bình phương của giá trị lớn nhất có thể ($1^2$, vì ảnh trong $[0, 1]$) |
| $10 \log_{10}$ | đổi ra decibel: MSE giảm 10 lần thì PSNR tăng 10 dB |

**Ví dụ:** tranh của dự án đạt khoảng 20.5 dB, tức $\text{MSE} = 10^{-2.05} \approx 0.0089$: mỗi con số lệch trung bình
khoảng $\sqrt{0.0089} \approx 0.094$ trên thang 0 đến 1.

**SSIM** (độ giống về cấu trúc), tính trên từng vùng nhỏ rồi lấy trung bình:

$$
\text{SSIM}(P, I) = \frac{(2 \mu_P \mu_I + C_1)(2 \sigma_{PI} + C_2)}{(\mu_P^2 + \mu_I^2 + C_1)(\sigma_P^2 + \sigma_I^2 + C_2)} .
$$

| Ký hiệu | Nghĩa | Giá trị |
| :--- | :--- | :--- |
| $\mu_P, \mu_I$ | độ sáng trung bình của vùng (cửa sổ Gauss $11 \times 11$, $\sigma = 1.5$) | |
| $\sigma_P^2, \sigma_I^2$ | phương sai: độ tương phản trong vùng | |
| $\sigma_{PI}$ | hiệp phương sai: hai vùng biến thiên có cùng chiều không (đường nét có khớp không) | |
| $C_1, C_2$ | hằng số nhỏ để không chia cho 0 | $(0.01)^2$ và $(0.03)^2$ |
| $\text{SSIM}$ | 1 là giống hệt | |

**LPIPS** (khác biệt theo cảm nhận), dùng các lớp của mạng AlexNet:

$$
\text{LPIPS}(P, I) = \sum_{l} \frac{1}{H_l W_l} \sum_{h, w} \Big\lVert \mathbf{w}_l \odot \big( \hat{f}^{\,l}_{hw}(P) - \hat{f}^{\,l}_{hw}(I) \big) \Big\rVert_2^2 .
$$

| Ký hiệu | Nghĩa |
| :--- | :--- |
| $l$ | một lớp của mạng AlexNet |
| $\hat{f}^{\,l}_{hw}(P)$ | đặc trưng (đã chuẩn hoá) của tranh tại lớp $l$, vị trí $(h, w)$ |
| $H_l, W_l$ | kích thước bản đồ đặc trưng của lớp $l$ |
| $\mathbf{w}_l$ | trọng số đã học để khớp với đánh giá của con người |
| $\lVert \cdot \rVert_2^2$ | bình phương độ dài vectơ |
| $\text{LPIPS}$ | càng **thấp** càng giống |

---

## Lưu ý

- Các công thức được viết lại từ mã nguồn và `report/chapters/02_theory.md`. Ví dụ bằng số là tính tay.
- Ở mục 7, các giá trị $\rho = 0.99$, $\varepsilon = 10^{-8}$ là mặc định của `torch.optim.RMSprop`; dự án không đổi chúng.
- SSIM và LPIPS chỉ dùng để chấm điểm, không tham gia vào việc vẽ. Khi vẽ, máy chỉ tối ưu độ sai L1 ở mục 7.
