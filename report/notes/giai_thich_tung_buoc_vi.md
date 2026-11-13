# Giải thích từng bước hoạt động của Neural Painter (ví dụ ảnh 900×600)

Tài liệu này đi theo đúng thứ tự chương trình chạy. Mỗi bước có cùng một khuôn: **làm gì**, **đầu vào và đầu ra**, **phép toán bên trong**,
**vì sao cần**, **mã nguồn**. Ví dụ xuyên suốt: một ảnh 900×600 (rộng 900, cao 600), cọ sơn dầu (`oil`), 500 nét, chế độ mặc định
`progressive`, mạng render nhẹ ("light") của nhóm tác giả 2021. Phần riêng về phép "stretch" nằm ở `stretch_resize_vi.md`; ở đây chỉ
nhắc lại kết quả.

Phương pháp gốc: Zou và cộng sự, *Stylized Neural Painting* (CVPR 2021). Mã của đề tài: gói `neural_painter/`.

---

## 0. Ý tưởng toàn cục

Chương trình **không "biết" phải vẽ gì**. Nó giải một bài toán tối ưu:

> Tìm danh sách các nét cọ $\theta = (\theta_1, \dots, \theta_N)$ sao cho bức tranh $C(\theta)$ ghép từ các nét đó gần ảnh gốc $I$ nhất.

$$\theta^\star = \arg\min_{\theta}\; \mathcal{L}\bigl(C(\theta),\, I\bigr)$$

Khó khăn: bộ vẽ chính xác (OpenCV, NumPy) không có đạo hàm, nên không thể dùng giảm gradient. Lời giải của bài báo: huấn luyện một mạng nơ-ron
(**neural renderer**) bắt chước bộ vẽ đó. Mạng này khả vi (differentiable) nên đạo hàm của sai số theo từng tham số nét tính được, và
ta "đẩy" các tham số theo hướng làm sai số giảm. Bộ vẽ chính xác chỉ được dùng ở cuối để vẽ bức tranh thật.

Sơ đồ tổng quát:

```
ảnh 900×600 → [1 đọc] → [2 chuẩn hoá] → [3 stretch 512×512] → [4 cắt khối theo mức lưới]
   → lặp 5 mức lưới (1×1 … 5×5):
        lặp từng nét trong khối:
             [5 đặt nét ở nơi sai nhiều nhất, lấy màu từ ảnh]
             lặp ~55 lần: [6 render] → [7 ghép alpha] → [8 loss] → [9 lan truyền ngược] → [10 cập nhật] → [11 kẹp giá trị]
        [12 đổi toạ độ khối → toàn ảnh] → [13 vẽ chính xác lên canvas 512×512] → canvas làm điểm xuất phát cho mức sau
   → [14 lưu tranh, trả hình dạng 512×341] → [15 time-lapse]
```

---

## 1. Đọc ảnh

- **Làm gì:** mở tệp, kiểm tra hợp lệ, áp dụng hướng xoay EXIF.
- **Vào:** tệp JPG, PNG, BMP, WebP hoặc TIFF. **Ra:** mảng số.
- **Bên trong:** từ chối định dạng khác, tệp hỏng, và ảnh trên 40 triệu điểm ảnh; trang web còn từ chối tệp tải lên trên 40 MB. Mỗi lỗi có
  thông báo cụ thể, và chương trình dừng **trước khi** tạo thư mục kết quả.
- **Vì sao:** ảnh điện thoại hay mang cờ xoay EXIF; nếu bỏ qua thì bức tranh bị nằm ngang. Giới hạn kích thước chặn việc hết bộ nhớ.
- **Mã:** `core/image_io.py` (`load_image`).

## 2. Chuẩn hoá

- **Làm gì:** đưa mọi ảnh về cùng một định dạng.
- **Vào:** ảnh xám, RGBA hoặc 8 bit. **Ra:** `float32` RGB, hình dạng $(600, 900, 3)$, giá trị trong $[0,1]$.
- **Bên trong:** chia cho 255; bỏ kênh alpha; nhân bản kênh xám ra 3 kênh; đảo thứ tự BGR của OpenCV thành RGB.
- **Vì sao:** toàn bộ phần sau giả định đúng một định dạng: `(H, W, 3)` dạng NumPy, `(1, 3, H, W)` dạng tensor. Thứ tự BGR của
  OpenCV không bao giờ rời khỏi bước đọc.
- **Mã:** `core/image_io.py` (`ensure_float01`, `to_tensor`).

## 3. Đưa về khung vuông 512×512 ("stretch")

- **Làm gì:** biến $(600, 900, 3)$ thành $(512, 512, 3)$.
- **Vì sao:** mạng render chỉ vẽ khối vuông, nên cần một khung vuông cố định.
- **Phép toán:** trung bình theo diện tích, viết dưới dạng tích ma trận cho mỗi kênh:
  $$T_c = R_y\, I_c\, R_x^{\top},\quad R_y \in \mathbb{R}^{512\times 600},\ R_x \in \mathbb{R}^{512\times 900},\quad
  R[a,b] = \frac{|[a s,(a+1)s)\cap[b,b+1)|}{s}$$
  với $s_x = 900/512 = 1{,}758$ và $s_y = 600/512 = 1{,}172$. Mỗi điểm ảnh đầu ra là trung bình các điểm ảnh nguồn mà nó phủ, điểm ở
  biên tính theo phần diện tích bị phủ. Đã kiểm chứng: khớp `cv2.INTER_AREA` đến $1{,}6\cdot10^{-7}$.
- **Hệ quả:** ngang bị nén $1{,}76$ lần, dọc $1{,}17$ lần, tức hình bị méo theo tỉ lệ $2/3$. Bộ tối ưu không biết điều này; hình dạng được
  trả lại ở bước 14.
- **Hai chế độ khác:** `crop` (cắt vuông $600\times600$ ở giữa rồi thu nhỏ, mất hai bên) và `pad` (thêm viền trắng cho vuông), đều không
  làm méo.
- **Mã:** `core/image_io.py` (`fit`, `resize`).

## 4. Chia thành các mức lưới và cắt khối

- **Làm gì:** chuẩn bị mục tiêu cho từng mức lưới $m = 1, \dots, 5$.
- **Vì sao 32×32:** mạng render nhẹ chỉ vẽ khối $32\times32$ điểm ảnh. Muốn vẽ chi tiết mịn thì phải vẽ **nhiều khối nhỏ**, mỗi khối là
  một phần của khung 512.
- **Phép toán:** với mỗi mức, đổi kích thước ảnh $512^2$ thành $32m \times 32m$ bằng cùng toán tử trung bình diện tích, rồi cắt thành
  $m^2$ khối $32\times 32$.

| Mức | Lưới | Thu nhỏ từ 512 xuống | Tensor mục tiêu | Mỗi khối phủ trên khung 512 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 1 × 1 | 32 × 32 | `(1, 3, 32, 32)` | cả khung (512 px) |
| 2 | 2 × 2 | 64 × 64 | `(4, 3, 32, 32)` | 256 px |
| 3 | 3 × 3 | 96 × 96 | `(9, 3, 32, 32)` | khoảng 171 px |
| 4 | 4 × 4 | 128 × 128 | `(16, 3, 32, 32)` | 128 px |
| 5 | 5 × 5 | 160 × 160 | `(25, 3, 32, 32)` | khoảng 102 px |

- **Ngân sách nét:** $1+4+9+16+25 = 55$ khối. $\lfloor 500/55 \rfloor = 9$ nét mỗi khối, nên thực tế vẽ $9 \times 55 = 495$ nét (5 nét dư
  bị bỏ, đúng theo mã gốc).
- **Xử lý theo lô:** tất cả khối của một mức đi qua mạng **cùng lúc** như một lô, nên chi phí trên GPU gần như không tăng theo số khối.
- **Mã:** `core/grid.py` (`grid_levels`, `strokes_per_block`, `img2patches`).

## 5. Đặt nét mới: ở đâu, to cỡ nào, màu gì

Mỗi khối lần lượt nhận 9 nét. Với mỗi nét mới, chương trình quyết định **điểm xuất phát** theo ba bước nhỏ.

### 5.1 Bản đồ sai số và chọn vị trí
- **Vào:** mục tiêu $I$ và canvas hiện tại $C$ của khối, cùng cỡ $(3, 32, 32)$.
- **Phép toán:**
  $$E = \Bigl(\operatorname{blur}_k\bigl(\textstyle\sum_{c} |I_c - C_c|\bigr)\Bigr)^{4},\qquad k = \lfloor 32/8 \rfloor = 4$$
  Tổng sai số theo 3 kênh màu, làm mờ bằng bộ lọc hộp $4\times4$ (gộp các điểm lỗi gần nhau), rồi **luỹ thừa 4** để chỗ sai rất nhiều
  nổi trội hơn hẳn chỗ sai ít.
- Bản đồ được phóng to song tuyến tính lên $512\times512$, chuẩn hoá thành xác suất $p = E/\sum E$, rồi **bốc ngẫu nhiên một điểm** theo $p$
  (tra hàm phân phối tích luỹ ngược). Đó là tâm $(c_x, c_y)$ của nét.
- **Trường hợp đặc biệt:** nếu canvas đã hoàn hảo ($E \equiv 0$) thì dùng bản đồ đều.
- **Vì sao:** nét mới đáng đặt ở nơi tranh còn sai nhất. Bốc ngẫu nhiên (thay vì luôn lấy điểm lớn nhất) giữ được tính đa dạng và tránh dồn
  mọi nét vào một chỗ.
- **Mã:** `pipeline/stroke_sampler.py` (`error_map`, `box_blur`, `StrokeSampler.positions`).

### 5.2 Màu xuất phát
- Màu RGB của ảnh mục tiêu tại điểm ảnh gần nhất với $(c_x, c_y)$ được **sao chép** làm màu của nét. Cọ hai màu (dầu, màu nước) nhận màu đó
  **hai lần**, nên hai đầu nét bắt đầu giống nhau.
- Đây là điểm xuất phát, **không phải kết quả cuối**: các số màu cũng là tham số và sẽ được tối ưu cùng các số khác (bước 10). Một điểm ảnh có
  thể là nhiễu hoặc nằm ở cạnh; nét phủ nhiều điểm ảnh và hoà lẫn với các nét bên dưới, nên bộ tối ưu kéo màu về giá trị làm cả bức tranh gần ảnh
  nhất. Với nét nằm trên vùng khá đều, đó là một màu tiêu biểu của vùng (hàm mất mát L1 ưu tiên trung vị hơn trung bình).

### 5.3 Các tham số còn lại (theo từng cọ)
Giá trị khởi tạo lấy ngẫu nhiên đều trong khoảng cấu hình ở mục `sampler` của `configs/<cọ>.yaml` (các khoảng này được chỉnh ở Tuần 6):

| Cọ | Tham số hình dạng | Cỡ ban đầu | Độ mờ đục $\alpha$ |
| :--- | :--- | :--- | :--- |
| Sơn dầu (12 số: 5/6/1) | $(c_x, c_y, w, h, \theta)$ | $w,h \sim U(0{,}2;\,0{,}4)$, $\theta \sim U(0,1)$ | $U(0{,}98;\,1)$ |
| Màu nước (15 số: 8/6/1) | 3 điểm điều khiển Bézier, 2 bán kính | cả 3 điểm đặt tại tâm, bán kính $U(0{,}2;\,0{,}4)$ | $U(0{,}5;\,0{,}7)$ |
| Bút dạ (12 số: 8/3/1) | 3 điểm điều khiển, 2 bán kính | đoạn thẳng dài 0,4 hướng ngẫu nhiên, bán kính $U(0{,}1;\,0{,}25)$ | $U(0{,}8;\,0{,}98)$ |
| Băng dính (9 số: 5/3/1) | $(c_x, c_y, w, h, \theta)$ | $w,h \sim U(0{,}2;\,0{,}4)$, $\theta = 0$ | $U(0{,}8;\,0{,}98)$ |

Mọi tham số nét nằm trong $[0,1]$. Điểm điều khiển giữa của đường Bézier được lưu **tương đối so với dây cung**, nên nét luôn hợp lệ dù
bộ tối ưu đẩy các tham số đi đâu.

- **Mã:** `core/procedural_rasterizer.py` (`initial_params_at`), `configs/*.yaml`.

---

## 6. Render bằng mạng nơ-ron

- **Làm gì:** vẽ **mọi nét đã đặt trong khối** (nét $0$ đến nét vừa đặt), mỗi nét thành một ảnh màu và một bản đồ trong suốt.
- **Vào:** $x = [\text{shape},\ \text{color},\ \alpha]$ của từng nét, ví dụ 12 số cho sơn dầu. **Ra:** với mỗi nét, ảnh màu $F_k$ và
  bản đồ alpha $A_k$ cỡ $32\times32$.
- **Mạng:** kiến trúc "fusion" của Zou và cộng sự. Nhánh hình dạng (kết nối đầy đủ rồi PixelShuffle) cho ra mặt nạ; nhánh màu (tích chập
  chuyển vị) cho ra ảnh màu. Tiền cảnh = màu × mặt nạ, alpha = độ mờ đục × mặt nạ.
- **Mạng bị đóng băng** (`requires_grad_(False)`): chỉ các nét được tối ưu, không bao giờ là mạng.
- **Hậu xử lý hình thái học:** tiền cảnh được **giãn** (cực đại trong cửa sổ $3\times3$), alpha được **co** (cực tiểu trong cửa sổ
  $3\times3$). Bước này bắt chước thao tác tương tự của bộ vẽ chính xác lúc suy luận, để ảnh mạng vẽ khớp ảnh bộ vẽ thật.
- **Mã:** `models/neural_renderer.py`, `core/morphology.py`, `PainterEngine.render` trong `pipeline/painter_engine.py`.

## 7. Ghép alpha các nét lên canvas

Nét $k$ được phủ lên canvas theo công thức alpha compositing:

$$C_k = A_k \odot F_k + (1 - A_k)\odot C_{k-1}$$

Nghĩa là: chỗ nét đục thì thấy màu nét, chỗ trong suốt thì thấy canvas cũ. Các nét ghép **theo thứ tự**, bắt đầu từ canvas xuất phát $C_0$
(canvas trắng cho sơn dầu ở mức 1; ở các mức sau là bức tranh các mức trước để lại, xem bước 13).

Mọi phép toán ở đây là phép nhân và cộng tensor nên đạo hàm đi ngược qua được:

$$\frac{\partial C_k}{\partial F_k} = A_k,\qquad \frac{\partial C_k}{\partial A_k} = F_k - C_{k-1},\qquad \frac{\partial C_k}{\partial C_{k-1}} = 1 - A_k.$$

- **Mã:** `core/differentiable_canvas.py` (`composite`, `composite_sequence`).

## 8. Hàm mất mát

So canvas cuối $C$ với khối mục tiêu $I$:

$$\mathcal{L} = \beta_1\,\frac{1}{N}\sum |C - I| \;+\; \beta_{OT}\,\mathcal{L}_{\text{Sinkhorn}}(C, I),\qquad \beta_1 = 1,\ \beta_{OT} = 0\ \text{(mặc định)}$$

Phần chính là **sai số tuyệt đối trung bình** (L1) trên điểm ảnh. Thành phần vận chuyển tối ưu (Sinkhorn) so sánh phân bố màu thay vì từng điểm
ảnh; nó tắt theo mặc định, giống mã gốc. Để theo dõi chất lượng, chương trình còn tính
$\text{PSNR} = 20\log_{10}\bigl(1/\sqrt{\text{MSE}}\bigr)$ (dB).

- **Mã:** `losses/pixel_loss.py`, `losses/sinkhorn.py`.

## 9. Lan truyền ngược

`loss.backward()` áp dụng quy tắc chuỗi từ $\mathcal{L}$ về từng tham số nét $\theta$:

$$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial C}\cdot\frac{\partial C}{\partial (F, A)}\cdot\frac{\partial (F, A)}{\partial \theta}$$

Thừa số đầu có dạng $\operatorname{sign}(C - I)/N$ (đạo hàm của L1). Thừa số giữa là các đạo hàm ở bước 7. Thừa số cuối là đạo hàm **qua mạng
render** (lý do mạng phải khả vi). Kết quả: với mỗi con số của mỗi nét, ta biết tăng nó một chút thì sai số tăng hay giảm bao nhiêu.

## 10. Cập nhật bằng RMSprop

Với mỗi tham số, gọi $g$ là gradient của nó. RMSprop giữ trung bình trượt của $g^2$ và chia bước đi cho căn của nó:

$$v \leftarrow \rho v + (1-\rho) g^2,\qquad \theta \leftarrow \theta - \eta\,\frac{g}{\sqrt{v} + \epsilon}$$

(với $\rho = 0{,}99$ và $\epsilon = 10^{-8}$ là giá trị mặc định của PyTorch). Chia cho $\sqrt v$ làm mọi tham số đi những bước cùng cỡ dù thang đo
gradient khác nhau (vị trí, màu và độ mờ đục có độ nhạy rất khác nhau).

**Tốc độ học theo đường cosine, khởi động lại ở mỗi nét mới:**

$$\eta_i = \eta_0\cdot\tfrac12\Bigl(1 + \cos\tfrac{\pi i}{T}\Bigr),\qquad \eta_0 = 0{,}004,\ i = 0,\dots,T-1,\ T = 55$$

Bước đầu đi xa (tìm tư thế đúng), bước cuối đi ngắn dần (tinh chỉnh). Khi đặt nét tiếp theo, $i$ quay về 0.

Quan trọng: bộ tối ưu cập nhật **tất cả nét đã đặt trong khối**, không chỉ nét mới. Nét cũ vì thế tiếp tục được điều chỉnh khi có nét mới
chồng lên.

## 11. Kẹp giá trị

Sau mỗi bước, các tham số hình dạng bị kẹp vào $[0{,}1;\,0{,}9]$ và màu cùng độ mờ đục vào $[0,1]$. Kẹp hình dạng ở $[0{,}1;\,0{,}9]$ để nét không bị
đẩy sát hoặc ra ngoài biên khối rồi biến mất; nét mất hẳn thì không còn gradient để quay lại.

## 11b. Lặp lại

Các bước 6 đến 11 lặp $T = \lfloor 500/9 \rfloor = 55$ lần cho **mỗi** nét. Mỗi mức lưới có $9 \times 55 = 495$ bước tối ưu, và 5 mức có tổng cộng
$5 \times 495 = 2475$ bước (số học từ tham số; không đo trực tiếp).

---

## 12. Đổi toạ độ khối sang toạ độ toàn ảnh

Sau khi một mức xong, các nét còn đang ở toạ độ **trong khối** $[0,1]^2$. Với khối ở hàng $r$, cột $c$ của lưới $m \times m$:

$$x \mapsto \frac{c}{m} + \frac{x}{m},\qquad y \mapsto \frac{r}{m} + \frac{y}{m},\qquad \text{kích thước} \mapsto \frac{\text{kích thước}}{m}$$

Điểm điều khiển Bézier tương đối được giữ nguyên. Ví dụ ở mức 5, khối số 7 là hàng 1, cột 2; một nét có $x = 0{,}5$ trong khối đó sẽ ở
$x = 2/5 + 0{,}5/5 = 0{,}5$ của cả khung.

**Thứ tự nét.** Nét thứ $j$ của mọi khối được xếp trước nét thứ $j+1$ của bất kỳ khối nào, và các khối được xáo trộn bằng bộ sinh số ngẫu nhiên
có hạt giống (seed). Nhờ vậy time-lapse lấp đầy cả bức tranh đều đặn thay vì hoàn thành từng khối.

- **Mã:** `core/grid.py` (`block_to_global`, `flatten_blocks`).

## 13. Vẽ chính xác và trao canvas cho mức sau

- Danh sách nét ở toạ độ toàn ảnh được vẽ **chồng lên canvas** bằng **bộ vẽ thủ tục chính xác** (`ProceduralRasterizer`, OpenCV và NumPy) ở độ
  phân giải $512\times512$. Đây là bản port từng điểm ảnh của `renderer.py` gốc, nên là ảnh thật chứ không phải ảnh mạng ước lượng.
- Canvas $512^2$ này được thu nhỏ bằng trung bình diện tích về $32(m+1)$ và cắt thành các khối của mức **kế tiếp**: đó là canvas xuất phát $C_0$
  của mức đó.
- **Vì sao đi từ thô đến mịn:** nét to ở mức thô sửa màu sắc và hình khối tổng thể; mức mịn chỉ thêm nét nhỏ ở những nơi sai số còn lớn,
  vì bản đồ sai số (5.1) luôn được tính trên canvas thừa hưởng từ mức trước. Nét thô vắt ngang biên các khối mịn, nên không để lại đường kẻ
  canvas trống dọc đường lưới.
- **Mã:** `pipeline/progressive_painter.py` (`paint_image`), `core/procedural_rasterizer.py`.

---

## 14. Lưu kết quả và trả hình dạng

- Danh sách nét cuối là mảng $(495, 12)$, giá trị trong $[0,1]$, lưu trong `strokes.npz` kèm tên cọ và màu canvas.
- Bức tranh $512\times512$ được đổi kích thước **chỉ theo chiều dọc** thành $512 \times 341$ (tỉ lệ $600/900$ nhân 512, làm tròn xuống), nên hình
  dạng 3:2 của ảnh gốc được trả lại; kích thước tuyệt đối thì không (không trở lại 900×600).
- **Các chỉ số** PSNR và SSIM được tính giữa bức tranh $512^2$ và ảnh đã fit `input.png`, ghi vào `results.csv`.

| Tệp | Nội dung |
| :--- | :--- |
| `final.png` | Bức tranh 512 × 341 |
| `input.png` | Ảnh gốc đã đưa về khung vuông (mục tiêu mà tranh được chấm điểm) |
| `strokes.npz` | Mọi nét theo thứ tự vẽ |
| `config.yaml` | Mọi thiết lập, thiết bị, mã commit |
| `results.csv` | PSNR, SSIM, thời gian, số nét, số bước, bộ nhớ |

## 15. Time-lapse

Time-lapse **không được ghi lúc đang vẽ**. Chương trình đọc `strokes.npz` và vẽ lại các nét theo thứ tự trên canvas trống, lưu MP4 (H.264) và
GIF. Nhờ vậy mọi bức tranh đã lưu đều có thể có time-lapse, và thứ tự ở bước 12 quyết định nó trông ra sao.

---

## Bảng theo dõi kích thước tensor (ảnh 900×600, sơn dầu)

| Giai đoạn | Hình dạng |
| :--- | :--- |
| Ảnh gốc sau chuẩn hoá | `(600, 900, 3)`, 1 620 000 số |
| Sau stretch | `(512, 512, 3)` |
| Mục tiêu mức $m$ | `(m², 3, 32, 32)` |
| Canvas xuất phát mức $m$ | `(m², 3, 32, 32)` |
| Tham số nét của mức $m$ | `(m², 9, 12)` (shape 5, màu 6, alpha 1) |
| Ảnh màu và alpha mạng vẽ khi đặt nét thứ $k+1$ | `(m²·(k+1), 3, 32, 32)` mỗi loại |
| Danh sách nét cuối | `(495, 12)` |
| Tranh vẽ chính xác | `(512, 512, 3)` |
| Tranh lưu | `(341, 512, 3)` |

## Những câu hỏi giáo sư có thể đặt ra

1. **Vì sao cần hai bộ vẽ?** Bộ vẽ chính xác không có đạo hàm; mạng render khả vi nhưng chỉ là ước lượng. Mạng dùng để *tìm* nét, bộ
   vẽ chính xác dùng để *vẽ* bức tranh cuối.
2. **Gradient descent có tìm được cực tiểu toàn cục không?** Không đảm bảo: bài toán không lồi. Điểm xuất phát (đặt nét theo bản đồ sai số,
   lấy màu từ ảnh) và lịch tốc độ học khởi động lại là các cách làm cho nghiệm tốt hơn trong thực tế, được đo bằng thí nghiệm loại trừ
   (ablation) ở chương kết quả của báo cáo.
3. **Vì sao dùng L1 mà không dùng L2?** L1 ít bị chi phối bởi vài điểm ảnh lệch rất xa và ưu tiên trung vị; đây cũng là lựa chọn của mã gốc.
4. **Vì sao 55 bước mỗi nét?** Đó là $\lfloor 500/9 \rfloor$, cách mã gốc chia: tổng ngân sách 500 bước chia đều cho 9 nét của một khối.
5. **Hạn chế lớn nhất?** Chi tiết bị mất ở bước đổi kích thước (ngang 900 xuống 160 ở mức 5), mạng chỉ vẽ khối $32\times32$, và với `stretch` hình
   dạng nét vẽ bị méo khi trả tỉ lệ ở bước 14 (suy luận, chưa đo trên tranh thật). `crop` và `pad` tránh được méo này.

## Phạm vi kiểm chứng

- **Đã kiểm tra bằng số:** dạng ma trận của phép đổi kích thước so với OpenCV; kích thước đầu ra 512×341; kích thước các tensor ở bước 3 và 4.
- **Đã đọc trong mã:** công thức bản đồ sai số, cách lấy màu, khoảng giá trị khởi tạo từng cọ, ghép alpha, hình thái học, kẹp giá trị, tốc độ học
  cosine, đổi toạ độ khối, việc trao canvas giữa các mức.
- **Lấy từ tài liệu dự án, không tự chạy lại:** kiến trúc mạng fusion; số liệu hiệu năng.
- **Chưa làm:** chạy trọn một bức tranh 900×600 để đối chiếu từng con số; $\rho = 0{,}99$, $\epsilon = 10^{-8}$ của RMSprop là giá trị mặc định của PyTorch và
  tôi không đọc lại mã nguồn của PyTorch trong phiên này.
