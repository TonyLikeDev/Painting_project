# Một tấm ảnh đi qua hệ thống như thế nào

Ghi chú này theo dõi một tấm ảnh từ lúc tải lên đến lúc thành tranh, với kích thước ma trận ở từng bước. Ảnh vào là
$A = [m, n]$ ($m$ hàng, $n$ cột); ví dụ số chạy song song là một ảnh điện thoại $m = 3000$, $n = 4000$, cọ sơn dầu,
500 nét, chế độ "Coarse to fine". Mã nguồn tương ứng: `neural_painter/core/image_io.py`, `core/grid.py`,
`pipeline/stroke_sampler.py`, `pipeline/painter_engine.py`, `pipeline/progressive_painter.py`, `app/gradio_app.py`.

![Tiền xử lý: cắt, co ảnh, chuyển thành tensor, chia lưới](../../figures/preprocessing_pipeline.png)

*Tiền xử lý ảnh: cắt về hình vuông, co về 512 px, chuyển thành tensor và chia lưới.*

## Tóm tắt kích thước

| Bước | Dữ liệu | Kích thước | Ví dụ |
| :--- | :--- | :--- | :--- |
| 1. Đọc file | $A$, số nguyên 0 đến 255 | $m \times n \times 3$ | $3000 \times 4000 \times 3$ |
| 2. Chuẩn hoá | $A$, số thực trong $[0, 1]$ | $m \times n \times 3$ | $3000 \times 4000 \times 3$ |
| 3. Đưa về vuông | $I$ | $512 \times 512 \times 3$ | $512 \times 512 \times 3$ |
| 4. Cắt khối (mức $g$) | $T_g$ | $g^2 \times 3 \times 32 \times 32$ | $25 \times 3 \times 32 \times 32$ ($g = 5$) |
| 5 đến 7. Tối ưu nét | $\theta$ | $g^2 \times 9 \times 12$ | $25 \times 9 \times 12$ |
| 8. Danh sách nét | nét | $495 \times 12$ | $495 \times 12$ |
| 9. Tranh | $P$ | $512 \times 512 \times 3$ | $512 \times 512 \times 3$ |
| 10. Kết quả | $P$, số nguyên 0 đến 255 | theo tỉ lệ $m : n$ | $384 \times 512 \times 3$ |

## 1. Đọc file thành ma trận $A$

$$
A \in \{0, 1, \dots, 255\}^{m \times n \times 3}, \qquad A[i, j] = (R, G, B),
$$

với $i$ là hàng ($0 \dots m-1$) và $j$ là cột ($0 \dots n-1$). $A$ thực chất là ba ma trận $R$, $G$, $B$, mỗi ma trận
$m \times n$.

- Ảnh điện thoại được xoay đúng chiều theo EXIF trước.
- Ảnh xám ($m \times n$) được chép thành 3 kênh; ảnh RGBA bị bỏ kênh trong suốt.
- Nếu $m \cdot n$ vượt quá 40 triệu pixel, ảnh bị từ chối.

## 2. Chuẩn hoá về $[0, 1]$

$$
A \leftarrow A / 255 \in [0, 1]^{m \times n \times 3}, \qquad (200, 120, 40) \mapsto (0.784, 0.471, 0.157).
$$

Mạng neural, hàm mất mát và PSNR $= 10 \log_{10}(1 / \mathrm{MSE})$ đều làm việc với giá trị trong $[0, 1]$.

## 3. Đưa về hình vuông $512 \times 512$, được ảnh đích $I$

Mặc định ("Keep the photo's shape"), ảnh được co thẳng, mỗi chiều một hệ số riêng:

$$
s_y = \frac{m}{512}, \qquad s_x = \frac{n}{512} \qquad (\text{ví dụ } 5.86 \text{ và } 7.81),
$$

$$
I[i, j] = \text{trung bình của } A \text{ trên ô } \big[i s_y, (i+1) s_y\big) \times \big[j s_x, (j+1) s_x\big).
$$

Mỗi pixel của $I$ là trung bình có trọng số (theo diện tích) của một ô $s_y \times s_x$ pixel gốc. Lấy trung bình
(`cv2.INTER_AREA`, giống bản gốc ở bước này) là lọc thông thấp trước khi giảm mẫu, nên chi tiết nhỏ không biến thành
hoa văn giả (răng cưa).

Hai cách còn lại:

- **Crop:** nếu $n > m$, cắt lấy các cột từ $(n - m)/2$ đến $(n + m)/2$, được $m \times m$, rồi mới co về 512.
- **Add borders:** thêm $(n - m)/2$ hàng trắng ở trên và ở dưới, được $n \times n$, rồi mới co về 512.

## 4. Cắt khối, được $T_g$

Hệ thống vẽ lần lượt 5 mức lưới $g = 1, 2, 3, 4, 5$. Ở mỗi mức, $I$ được co về $32g \times 32g$ (lại lấy trung bình) và
cắt thành $g^2$ khối $32 \times 32$, đánh số theo hàng. Ở lần co này, bản gốc dùng co song tuyến, không lấy trung bình;
dự án đổi sang lấy trung bình và đo được tranh tốt hơn $+0.32$ dB (báo cáo, Bảng 4.14).

$$
T_g[k, c, y, x] = I_g\big[c,\; 32 \lfloor k / g \rfloor + y,\; 32 (k \bmod g) + x\big], \qquad
T_g \in [0, 1]^{g^2 \times 3 \times 32 \times 32}.
$$

| $g$ | Số khối | Mỗi khối phủ trên ảnh 512 |
| :---: | :---: | :---: |
| 1 | 1 | cả ảnh |
| 2 | 4 | $256 \times 256$ px |
| 3 | 9 | $171 \times 171$ px |
| 4 | 16 | $128 \times 128$ px |
| 5 | 25 | $102 \times 102$ px |

Kích thước 32 là vì bộ vẽ neural "nhẹ" chỉ xuất ảnh $32 \times 32$. $g^2$ khối được đưa vào GPU cùng lúc như một batch.

## 5. Canvas ban đầu

Canvas $C$ có cùng kích thước $g^2 \times 3 \times 32 \times 32$:

- **Mức 1:** toàn 1 (trắng, cho sơn dầu và màu nước) hoặc toàn 0 (đen, cho bút dạ và băng dính).
- **Mức $g > 1$:** tranh các mức trước đã vẽ, co và cắt khối giống bước 4.

## 6. Tìm chỗ đặt nét mới

$$
E = \sum_{c \in \{R, G, B\}} |T_c - C_c| \in \mathbb{R}^{g^2 \times 32 \times 32}, \qquad
p = \frac{\big(\operatorname{blur}_4(E)\big)^4}{\sum \big(\operatorname{blur}_4(E)\big)^4}.
$$

Bốc ngẫu nhiên một điểm $(y, x)$ theo $p$. Nét mới đặt tâm tại đó, với màu ban đầu $T[k, :, y, x]$, tức màu ảnh gốc ngay
chỗ đó. Lũy thừa 4 làm nét dồn vào nơi sai nhiều nhất: vùng sai gấp đôi được chọn nhiều hơn $2^4 = 16$ lần.

## 7. Tối ưu: ảnh đứng yên, nét thay đổi

$$
\theta \in [0, 1]^{g^2 \times 9 \times 12} \quad (\text{9 nét mỗi khối, 12 số mỗi nét với cọ dầu}),
$$

$$
C = \text{ghép các nét do mạng neural vẽ}, \qquad
\mathcal{L} = \operatorname{mean} |C - T| \;\rightarrow\; \nabla_\theta \mathcal{L} \;\rightarrow\; \text{cập nhật } \theta .
$$

Từ bước này, ma trận ảnh $T$ không đổi nữa: nó chỉ là đáp án để so sánh.

## 8. Đổi nét về toạ độ cả ảnh

Nét ở khối hàng $r$, cột $c$ của lưới $g \times g$:

$$
x = \frac{c + x_{\text{khối}}}{g}, \qquad y = \frac{r + y_{\text{khối}}}{g}, \qquad
\text{kích thước} = \frac{\text{kích thước}_{\text{khối}}}{g}.
$$

Ví dụ khối $(r, c) = (2, 3)$ của lưới $5 \times 5$, $x_{\text{khối}} = y_{\text{khối}} = 0.5$: nét nằm ở
$(x, y) = (0.7, 0.5)$, tức pixel $(358, 256)$ trên ảnh 512. Gom tất cả các mức lại được danh sách $495 \times 12$
($9 \times (1 + 4 + 9 + 16 + 25) = 495$).

## 9. Vẽ tranh bằng bộ vẽ thật

Bộ vẽ thủ tục vẽ lần lượt 495 nét lên canvas $512 \times 512 \times 3$:

$$
P \leftarrow \hat{A}_k \odot F_k + (1 - \hat{A}_k) \odot P .
$$

## 10. Trả về đúng tỉ lệ ảnh gốc

$$
P \rightarrow
\begin{cases}
\lfloor 512\, m / n \rfloor \times 512 & \text{nếu } m < n \text{ (ảnh ngang, ví dụ } 384 \times 512\text{)},\\
512 \times \lfloor 512\, n / m \rfloor & \text{nếu } m \ge n \text{ (ảnh dọc)},
\end{cases}
\qquad
P \leftarrow \lfloor 255\, P + 0.5 \rfloor \in \{0, \dots, 255\}.
$$

Kết quả được lưu thành `final.png`, `final.jpg`, `input.png` (ảnh $512 \times 512$ đã đưa vào hệ thống) và
`strokes.npz` (danh sách nét).

## Lưu ý

- Dù ảnh vào lớn đến đâu, tranh ra chỉ dài 512 px ở cạnh dài; chi tiết nhỏ hơn khoảng 1/512 chiều ảnh đã mất từ bước 3.
- Ở chế độ mặc định, tranh được vẽ trên ảnh đã bị ép vuông rồi mới kéo về tỉ lệ $m : n$, nên nét cọ trong ảnh cuối hơi
  bị bóp méo. Chọn "Crop" thì tránh được việc này, đổi lại mất hai mép ảnh.
- Các con số ví dụ (5.86, 7.81, pixel $(358, 256)$, $384 \times 512$) được tính tay từ mã nguồn, chưa chạy với một ảnh
  $3000 \times 4000$ thật.
