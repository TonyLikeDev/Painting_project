# Chuẩn hoá ảnh về $[0, 1]$

Ghi chú này giải thích bước 2 trong [`image_processing_vi.md`](image_processing_vi.md): ảnh vừa đọc từ file được đưa
từ số nguyên 0 đến 255 về số thực trong $[0, 1]$. Mã nguồn: `neural_painter/core/image_io.py` (`load_image`,
`ensure_float01`, `to_uint8`).

## 1. Phép biến đổi

Chuẩn hoá chỉ là một phép chia cho từng phần tử của ma trận ảnh $A = [m, n]$ (`image_io.py:67`):

$$
A_{\text{mới}}[i, j, c] = \frac{A[i, j, c]}{255}, \qquad c \in \{R, G, B\}.
$$

| | Trước | Sau |
| :--- | :--- | :--- |
| Kiểu số | `uint8` (số nguyên) | `float32` (số thực) |
| Giá trị | 256 mức: $0, 1, \dots, 255$ | trong $[0, 1]$, hai mức liền nhau cách $1/255 \approx 0.0039$ |
| Kích thước | $m \times n \times 3$ | $m \times n \times 3$ (không đổi) |

Ví dụ:

$$
(200, 120, 40) \mapsto (0.784, 0.471, 0.157), \qquad 0 \mapsto 0 \text{ (đen)}, \qquad 255 \mapsto 1 \text{ (trắng)}.
$$

Đây là phép nhân với một hằng số dương ($1/255$), không cộng trừ gì, nên:

- tỉ lệ giữa các pixel giữ nguyên: pixel sáng gấp đôi vẫn sáng gấp đôi;
- không mất thông tin: nhân lại 255 là ra đúng ảnh cũ;
- thứ tự sáng tối giữ nguyên: $a < b \Rightarrow a/255 < b/255$.

## 2. Các trường hợp đặc biệt

Quy tắc chung: chia cho giá trị lớn nhất mà định dạng lưu được, tức $2^b - 1$ với $b$ là số bit mỗi kênh.

| Ảnh vào | Xử lý | Mã |
| :--- | :--- | :--- |
| RGB 8 bit (JPG, PNG thường) | chia $255 = 2^8 - 1$ | `image_io.py:67` |
| RGBA | bỏ kênh trong suốt, rồi chia 255 | `image_io.py:64` |
| Xám 8 bit | chép thành 3 kênh, rồi chia 255 | `image_io.py:64` |
| Xám 16 bit (PNG, TIFF) | chia $65535 = 2^{16} - 1$, rồi chép thành 3 kênh | `image_io.py:61-63` |
| Mảng numpy (ảnh dán từ app) | `uint8`: chia 255; số thực: kẹp về $[0, 1]$ | `ensure_float01`, `image_io.py:83-96` |

Ảnh 16 bit cần nhánh riêng. Nếu chuyển thẳng sang RGB 8 bit, mọi giá trị trên 255 (trong số 65535 mức) bị cắt
xuống 255 và gần như cả ảnh thành trắng. Chia cho 65535 thì ảnh 16 bit và ảnh 8 bit của cùng một cảnh cho cùng
kết quả trong $[0, 1]$.

Trước phép chia, ảnh đã được xoay đúng chiều theo EXIF và kiểm tra giới hạn 40 triệu pixel (`image_io.py:58-60`).

## 3. Vì sao phải đưa về $[0, 1]$

### 3.1. Cùng thang với bộ vẽ

Bộ vẽ neural chỉ xuất được giá trị trong $[0, 1]$. Tham số nét luôn nằm trong $[0, 1]$, nên màu nét $c \in [0, 1]$.
Mặt nạ $m \in [0, 1]$, và nét vẽ ra là

$$
\hat F = c \odot m \in [0, 1]^{3 \times 32 \times 32}.
$$

Hàm mất mát so trực tiếp canvas $C$ với ảnh đích $T$:

$$
\mathcal{L} = \operatorname{mean} |C - T| .
$$

Nếu $T$ vẫn ở thang 0 đến 255, canvas không bao giờ vượt quá 1. Mọi pixel có $T > 1$ đều đòi "sáng hơn nữa", bộ tối
ưu đẩy mọi nét về màu sáng nhất, và tranh ra gần như toàn trắng.

### 3.2. Ghép nét không bao giờ tràn

Mỗi nét được ghép lên canvas theo

$$
C_k = A_k \odot F_k + (1 - A_k) \odot C_{k-1}, \qquad A_k \in [0, 1].
$$

Với mỗi pixel, đây là trung bình có trọng số của hai số $F_k$ và $C_{k-1}$, nên kết quả nằm giữa hai số đó:

$$
\min(F_k, C_{k-1}) \le C_k \le \max(F_k, C_{k-1}).
$$

Canvas bắt đầu ở $C_0 = 1$ (trắng) hoặc $C_0 = 0$ (đen), mọi $F_k \in [0, 1]$. Áp dụng lần lượt cho từng nét
(quy nạp theo $k$) thì $C_k \in [0, 1]$ với mọi $k$, không cần cắt giá trị ở bất kỳ bước nào.

### 3.3. Màu khởi tạo lấy thẳng từ ảnh

Ở bước 6 của [`image_processing_vi.md`](image_processing_vi.md), nét mới nhận màu ban đầu $T[k, :, y, x]$, tức màu ảnh gốc tại chỗ đặt nét.
Phép gán này chỉ đúng khi giá trị pixel và tham số màu cùng thang $[0, 1]$.

### 3.4. PSNR với $\text{MAX} = 1$

$$
\text{PSNR} = 10 \log_{10} \frac{\text{MAX}^2}{\text{MSE}} \;\xrightarrow{\text{MAX} = 1}\; 10 \log_{10} \frac{1}{\text{MSE}} .
$$

Ở thang 0 đến 255 thì MAX = 255 và MSE lớn gấp $255^2$ lần, nên ra đúng con số đó. PSNR không phụ thuộc thang đo,
miễn MAX khớp với thang.

### 3.5. Phải là số thực

- Gradient cần giá trị liên tục; số nguyên không có đạo hàm.
- Phép co ảnh lấy trung bình (bước 3) tạo ra số lẻ: trung bình của 200 và 201 là 200.5, `uint8` sẽ làm tròn.
- `uint8` tràn số: $200 + 100$ ra $44$ (tính theo modulo 256).
- `float32` có khoảng 7 chữ số có nghĩa, mịn hơn rất nhiều so với bước $1/255$.

### 3.6. Lý do *không phải* là "số lớn thì mạng học kém"

Bộ tối ưu RMSprop chia gradient cho độ lớn trung bình của chính nó:

$$
\theta \leftarrow \theta - \eta \, \frac{g}{\sqrt{v} + \varepsilon}, \qquad v \approx \operatorname{mean}(g^2).
$$

Nhân hàm mất mát với 255 thì $g$ và $\sqrt{v}$ cùng lớn gấp 255 lần, nên bước cập nhật gần như không đổi. Vấn đề
thật là lệch thang giữa ảnh đích và những gì bộ vẽ xuất ra được (mục 3.1), không phải độ lớn của con số.

## 4. Những gì bước này không làm

- **Không trừ trung bình, chia độ lệch chuẩn** kiểu ImageNet $(x - \mu)/\sigma$ như các mạng phân loại ảnh. Việc
  đó không cần vì mạng neural ở đây không nhận ảnh làm đầu vào: nó nhận tham số nét (12 số với cọ dầu) và xuất ra
  ảnh. Ảnh gốc chỉ là đáp án để so. File `models/vgg_perceptual.py` có chuẩn hoá ImageNet, nhưng pipeline vẽ không
  dùng file này.
- **Không đổi gamma.** Giá trị vẫn là sRGB, nên $0.5$ không có nghĩa là "một nửa lượng ánh sáng" về mặt vật lý
  (giá trị ánh sáng tuyến tính tương ứng khoảng $0.21$). Ghép nét và hàm mất mát đều tính trên sRGB, giống bản gốc
  và giống dữ liệu bộ vẽ được huấn luyện, nên giữ nguyên là đúng.

## 5. Chiều ngược lại khi lưu tranh

`to_uint8` (`image_io.py:99-103`) đưa tranh $P$ về số nguyên trước khi lưu PNG/JPG:

$$
P_{8\text{bit}} = \big\lfloor 255 \cdot \operatorname{clip}(P, 0, 1) + 0.5 \big\rfloor ,
$$

tức làm tròn tới số gần nhất. Sai số tối đa $0.5/255 \approx 0.002$, mắt không thấy được. Một pixel đi trọn vòng
$x \to x/255 \to \lfloor 255 \cdot x/255 + 0.5 \rfloor$ trở lại đúng $x$.

## Lưu ý

Các ý trên được suy ra từ mã nguồn, chưa chạy thử. Hai khẳng định chưa đo bằng thực nghiệm: "tranh ra gần như toàn
trắng" ở mục 3.1 và "bước cập nhật gần như không đổi" ở mục 3.6.
