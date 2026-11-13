# Phép "stretch" biến ảnh 900×600 thành khung vuông 512×512

Ghi chú này giải thích, bằng ngôn ngữ toán học, cách hệ thống biến một ảnh chữ nhật (ví dụ 900×600 điểm ảnh) thành khung vuông
512×512 trước khi vẽ, rồi trả lại hình dạng ban đầu khi lưu tranh. Mã nguồn liên quan: `neural_painter/core/image_io.py`
(`fit`, `resize`, `preprocess`, `PreprocessedImage.output_size`) và `neural_painter/core/grid.py` (`img2patches`, `block_to_global`).

## 1. Bài toán

- Ảnh gốc là một tensor $I \in [0,1]^{H \times W \times 3}$ với $H = 600$ (chiều cao) và $W = 900$ (chiều rộng), tức
  $600 \times 900 \times 3 = 1\,620\,000$ số thực.
- Mạng render thần kinh (neural renderer) chỉ làm việc trên các khối vuông, nên toàn bộ quy trình cần **một khung vẽ vuông cố định**
  cạnh $S = 512$.
- Chính sách `stretch` (hành vi của mã gốc, và là mặc định của cả `run_paint.py` lẫn ứng dụng web) chỉ **thay đổi kích thước thuần
  túy** và không quan tâm đến tỉ lệ khung hình của ảnh.

## 2. Phép đổi kích thước là một tích ma trận

Với một kênh màu $I_c$, ảnh sau khi "stretch" là

$$T_c = R_y \, I_c \, R_x^{\top}, \qquad R_y \in \mathbb{R}^{512 \times 600},\quad R_x \in \mathbb{R}^{512 \times 900}.$$

Hệ số tỉ lệ theo hai trục là $s_x = 900/512 = 1{,}758$ và $s_y = 600/512 = 1{,}172$. Điểm ảnh đầu ra thứ $a$ phủ đoạn
$[a s, (a+1)s)$ trên trục nguồn, còn điểm ảnh nguồn thứ $b$ chiếm đoạn $[b, b+1)$. Trọng số là

$$R[a,b] = \frac{\bigl|\,[a s,(a+1)s) \cap [b, b+1)\,\bigr|}{s}.$$

Đây chính là quy tắc **trung bình theo diện tích** (bộ lọc hộp, `cv2.INTER_AREA`): mỗi điểm ảnh đầu ra là trung bình của hình chữ nhật
điểm ảnh nguồn mà nó phủ, các điểm ảnh ở biên được tính theo phần diện tích bị phủ.

**Kiểm chứng bằng số.** Tôi dựng tường minh $R_y$ và $R_x$ rồi so $R_y I R_x^{\top}$ với `cv2.resize(..., INTER_AREA)` trên dữ liệu
ngẫu nhiên. Sai khác lớn nhất là $1{,}6 \times 10^{-7}$, tức bằng độ chính xác của số thực 32 bit.

Các tính chất suy ra trực tiếp từ cách xây dựng:

- **Tổ hợp lồi.** Mỗi hàng của $R$ không âm và có tổng bằng 1 (đã kiểm tra), nên giá trị đầu ra luôn nằm trong $[0,1]$, không có hiện
  tượng vọt lố (overshoot) hay gợn sóng (ringing).
- **Bảo toàn giá trị trung bình.** Mỗi điểm ảnh nguồn được phủ trọn nên tổng trọng số của nó là $1/s$; lấy tổng rồi chia cho số điểm ảnh
  đầu ra cho $\bar T = \bar I$.
- **Thưa.** Mỗi hàng của $R_x$ chỉ có 2 đến 3 phần tử khác không (đã kiểm tra).
- **Tách được theo trục.** Hàng và cột được xử lý độc lập; chi phí chỉ cỡ $O(HW)$.

## 3. Hệ quả hình học

Bước điểm ảnh thay đổi theo từng trục: $a_x = 512/900 = 0{,}569$ và $a_y = 512/600 = 0{,}853$. Độ bất đẳng hướng (anisotropy) là
$a_x / a_y = H/W = 2/3$: cấu trúc theo chiều ngang bị nén nhiều hơn chiều dọc 1,5 lần. Bộ tối ưu chỉ nhìn thấy ảnh đã méo này và
**không biết** nó đã bị méo.

## 4. Chia thành các khối

Với mỗi mức lưới $m \in \{1, \dots, 5\}$, quy trình lại đổi kích thước ảnh $512^2$ thành $32m \times 32m$ bằng cùng toán tử (hệ số
thu nhỏ lần lượt 16, 8, 5,33, 4 và 3,2), rồi cắt thành $m^2$ khối $32 \times 32$ (kích thước mà mạng render vẽ được). Tất cả các khối
đi vào mạng như **một lô (batch)** duy nhất, ví dụ `(25, 3, 32, 32)` ở mức 5.

| Mức | Lưới | Kích thước sau khi thu nhỏ | Tensor các khối | Hệ số thu nhỏ từ 512 |
| :--- | :--- | :--- | :--- | ---: |
| 1 | 1 × 1 | 32 × 32 | `(1, 3, 32, 32)` | 16 |
| 2 | 2 × 2 | 64 × 64 | `(4, 3, 32, 32)` | 8 |
| 5 | 5 × 5 | 160 × 160 | `(25, 3, 32, 32)` | 3,2 |

Ở mức mịn nhất, mỗi điểm ảnh mục tiêu là trung bình của khoảng $5{,}6 \times 3{,}75$ điểm ảnh của ảnh gốc 900×600. Hai bộ lọc hộp liên
tiếp cho kết quả hơi mượt hơn một bộ lọc hộp trực tiếp; đây là một xấp xỉ được chấp nhận.

## 5. Trả nét vẽ về toạ độ của toàn ảnh

Các nét vẽ được tối ưu trong toạ độ của từng khối, $[0,1]^2$. Với khối ở hàng $r$, cột $c$, hàm `block_to_global` ánh xạ

$$x \mapsto \frac{c}{m} + \frac{x}{m}, \qquad y \mapsto \frac{r}{m} + \frac{y}{m},$$

và kích thước nét thành $\text{size}/m$ (đã đọc trong mã nguồn). Sau đó bộ vẽ thủ tục chính xác (procedural rasterizer) vẽ lại toàn
bộ nét lên khung $512 \times 512$.

## 6. Hoàn trả hình dạng khi lưu tranh

`output_size` lấy tỉ lệ $\text{aspect} = H/W = 600/900$ và trả về $(512, \lfloor 512 \cdot 2/3 \rfloor) = (512, 341)$. Bức tranh
$512 \times 512$ được đổi kích thước lần nữa bằng cùng toán tử, **chỉ theo chiều dọc**: $T' = R_{341 \times 512}\, P$, trong đó $P$ là
bức tranh vuông.

- **Tỉ lệ tổng hợp:** theo chiều ngang $512/900 = 0{,}5689$, theo chiều dọc $341/600 = 0{,}5683$. Hai giá trị khớp nhau đến 0,1%; phần
  dư đến từ việc làm tròn xuống $341{,}33 \to 341$.
- **Kích thước:** ảnh lưu có kích thước **512 × 341**, không phải 900 × 600. Hình dạng được trả lại nhưng độ phân giải thì không.

## 7. Hạn chế cần nêu khi bảo vệ

- **Mất thông tin.** Chiều ngang bị thu từ 900 xuống 512, và xuống 160 ở mức lưới 5. Chi tiết mịn của ảnh gốc mất đi trước khi bộ vẽ
  nhìn thấy ảnh; kích thước khối 32×32 là một giới hạn thứ hai.
- **Hình dạng nét vẽ bị méo khi lưu (suy luận, chưa kiểm chứng trên tranh thật).** Các nét được vẽ trong khung đã bị nén rồi cả bức
  tranh mới được lấy mẫu lại, nên hình dạng và góc của nét bị co không đều: hướng $(\cos\theta, \sin\theta)$ thành
  $(\cos\theta, \tfrac{2}{3}\sin\theta)$ trong ảnh lưu, và kết cấu cọ bị ép dẹt. Hai chế độ `crop` (cắt giữa, mất hai bên) và `pad`
  (thêm viền trắng) tránh được méo này.
- **Răng cưa (aliasing) của bộ lọc hộp.** Bộ lọc hộp không phải bộ lọc thông thấp lý tưởng, nhưng vẫn tốt hơn phép thu nhỏ song tuyến
  tính (bilinear) của mã gốc, vốn không lấy trung bình.

## Phạm vi kiểm chứng

Đã kiểm tra bằng số: dạng ma trận của phép đổi kích thước so với OpenCV, kích thước đầu ra (512, 341), các hệ số thu nhỏ theo mức lưới.
Đã đọc trong mã: công thức `block_to_global`. Chưa đo: mức độ méo của nét vẽ trên một bức tranh thật (mục 7 là suy luận).
