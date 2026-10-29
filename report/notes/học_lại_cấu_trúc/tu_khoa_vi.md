# Từ khoá của bài toán

Ghi chú này giải thích dự án qua các **từ khoá** (keyword), xếp theo đúng thứ tự một tấm ảnh đi qua hệ thống. Mỗi từ
khoá gồm tên tiếng Anh (thường gặp trong tài liệu và mã), nghĩa tiếng Việt, giải thích ngắn và chỗ dùng trong dự án.
Giải thích dài hơn có ví dụ: [`how_it_works_simple_vi.md`](how_it_works_simple_vi.md).

## Bài toán trong một câu

> Cho một **ảnh** (image), tìm một danh sách **nét cọ** (strokes) sao cho khi **vẽ** (render) các nét đó lên
> **canvas** thì được bức tranh **giống ảnh nhất**. Độ giống được đo bằng một **hàm mất mát** (loss), và các nét được
> chỉnh dần bằng **gradient** nhờ một **mạng neural bắt chước cọ vẽ** (neural renderer).

## Mục lục

| Nhóm | Từ khoá |
| :--- | :--- |
| [0. Tổng quan](#0-tổng-quan) | Stroke-based rendering, Optimization-based, Stylized Neural Painting |
| [1. Ảnh đầu vào](#1-ảnh-đầu-vào) | Pixel, RGB, Tensor, Normalization, Resize, Area averaging, Aliasing, Aspect ratio |
| [2. Lưới](#2-lưới) | Grid, Block, Coarse-to-fine, Batch |
| [3. Cọ vẽ](#3-cọ-vẽ) | Stroke, Stroke parameters, Brush, Texture, Canvas, Opacity, Compositing, Rasterizer |
| [4. Mạng neural](#4-mạng-neural) | Differentiable, Neural renderer, Surrogate, FusionNet, Light renderer, Training, Epoch, Fidelity, Dilate / erode |
| [5. Tối ưu](#5-tối-ưu) | Loss, Gradient, Backpropagation, Gradient descent, Optimizer, Learning rate, Cosine schedule, Iteration, Clamp, Error map, Sampling, Allocation, Optimal transport |
| [6. Đánh giá](#6-đánh-giá) | PSNR, SSIM, LPIPS, Seed, Ablation |
| [7. Đầu ra](#7-đầu-ra) | Stroke file, Time-lapse, Web app, Device, Speed setting |

---

## 0. Tổng quan

**Stroke-based rendering (SBR), vẽ bằng nét.** Tạo ra một hình bằng cách đặt chồng nhiều nét cọ lên nhau, thay vì
tô từng pixel. Kết quả trông như tranh vẽ tay vì nó *được làm* từ nét cọ.

**Optimization-based, dựa trên tối ưu.** Dự án không có mạng neural nào "học vẽ tranh" từ một bộ tranh mẫu. Với
mỗi ảnh mới, máy tính *giải một bài toán tối ưu*: chỉnh các con số của nét cọ cho tới khi tranh giống ảnh. Không có
dữ liệu tranh nào được dùng ở bất kỳ bước nào.

**Stylized Neural Painting.** Bài báo gốc (Zou và cộng sự, CVPR 2021) mà dự án viết lại và hiện đại hoá: PyTorch 2,
chạy trên `cuda` / `mps` / `cpu`, giao diện web, xuất video time-lapse.

---

## 1. Ảnh đầu vào

**Pixel, điểm ảnh.** Ô vuông màu nhỏ nhất của ảnh. Ảnh $3000 \times 4000$ có 12 triệu pixel.

**RGB, kênh màu.** Mỗi pixel gồm 3 số: đỏ (R), xanh lá (G), xanh dương (B). Trong file ảnh, mỗi số nằm trong 0..255.
Ví dụ $(200, 120, 40)$ là màu cam nâu.

**Tensor.** Mảng nhiều chiều mà PyTorch tính toán trên GPU. Ảnh trong mã có dạng `(1, 3, H, W)`: 1 ảnh, 3 kênh,
$H$ hàng, $W$ cột. Ảnh dạng numpy là `(H, W, 3)`.

**Normalization, chuẩn hoá.** Chia mỗi số cho 255 để đưa về $[0, 1]$, cùng thang với màu cọ.
`core/image_io.py:67`. Chi tiết: [`normalization_vi.md`](normalization_vi.md).

**Resize / downsampling, thu nhỏ ảnh.** Đưa ảnh về $512 \times 512$, rồi ở mỗi vòng về $32g \times 32g$. Ít pixel
hơn thì mỗi lần "vẽ thử rồi so" nhanh hơn.

**Area averaging (`cv2.INTER_AREA`), lấy trung bình theo diện tích.** Mỗi pixel mới là **trung bình cộng (mean)**
của các pixel cũ nằm trong vùng của nó. Pixel nằm một phần trong vùng thì được tính theo phần đó. Ở lần co về các ô,
dự án dùng cách này thay cho co song tuyến của bài gốc, và đo được tranh tốt hơn $+0.32$ dB.

**Aliasing, răng cưa.** Chi tiết nhỏ biến thành hoa văn giả khi thu nhỏ mà không lấy trung bình. Area averaging
tránh được hiện tượng này.

**Aspect ratio, tỉ lệ khung hình.** Ảnh $3000 \times 4000$ có tỉ lệ $3 : 4$, không vuông. Hệ thống có ba cách đưa về
hình vuông: *stretch* (ép, mặc định), *crop* (cắt hai mép), *pad* (thêm viền trắng). Tranh cuối được trả về đúng tỉ lệ
gốc, ví dụ $384 \times 512$.

---

## 2. Lưới

**Grid, lưới.** Chia ảnh thành $g \times g$ ô. Lý do: mạng neural chỉ vẽ được một ô $32 \times 32$ pixel
(`core/grid.py`).

**Block / patch, ô.** Một ô của lưới, luôn được đưa về $32 \times 32$. Mỗi ô có các nét riêng. Toạ độ nét trong ô
được đổi ra toạ độ cả ảnh: $x_{\text{ảnh}} = (\text{cột} + x_{\text{ô}}) / g$.

**Coarse-to-fine / progressive, từ thô đến mịn.** Vẽ lần lượt các lưới $1 \times 1, 2 \times 2, \dots, 5 \times 5$:
nét to phủ nền trước, nét nhỏ tả chi tiết sau. Tổng cộng $9 \times 55 = 495$ nét. Đây là chế độ mặc định ("Coarse to
fine"). Hai chế độ khác là *fixed* (chỉ lưới $5 \times 5$) và *full* (chỉ một ô cho cả ảnh).

| Chế độ | PSNR (dB) | Thời gian (s) |
| :--- | ---: | ---: |
| progressive | 20.55 | 12.1 |
| fixed | 20.31 | 5.7 |
| full | 13.82 | 30.8 |

*Cọ sơn dầu, 500 nét, GPU RTX 3070 (báo cáo, Bảng 4.29).*

**Batch, lô.** Mọi ô của một vòng được đưa vào GPU **cùng lúc** như một lô: vòng $5 \times 5$ vẽ 25 ô song song.

---

## 3. Cọ vẽ

**Stroke, nét cọ.** Một vệt màu đặt lên tranh. Với máy tính, một nét chỉ là một dãy số.

**Stroke parameters ($\theta$), tham số nét.** Dãy số mô tả một nét, mỗi số trong $[0, 1]$. Cọ sơn dầu có 12 số:
vị trí tâm $(x_c, y_c)$, kích thước $(w, h)$, góc $\theta$, màu đầu $(r_0, g_0, b_0)$, màu cuối $(r_2, g_2, b_2)$, độ
trong suốt $\alpha$. Cả bức tranh là một bảng $495 \times 12$ số.

**Brush / material, loại cọ.** Bốn loại: sơn dầu (12 số), màu nước (15), bút dạ (12), băng dính màu (9). Mỗi loại khai
báo trong một file `neural_painter/configs/*.yaml`.

**Brush texture, mẫu vệt cọ.** Ảnh chụp vệt cọ thật (`neural_painter/assets/brushes/`), được tô màu, kéo giãn, xoay
và đặt vào vị trí để thành một nét sơn dầu.

**Canvas, nền vẽ.** Tấm tranh đang vẽ. Ban đầu toàn trắng (sơn dầu, màu nước) hoặc toàn đen (bút dạ, băng dính).

**Opacity / alpha ($A$), độ che phủ.** Với mỗi pixel, nét che bao nhiêu phần màu bên dưới: 1 là che kín, 0 là không
che.

**Compositing / alpha blending, ghép nét.** Dán từng nét lên canvas theo thứ tự:

$$
C_k = A_k \odot F_k + (1 - A_k) \odot C_{k-1},
$$

với $F_k$ là màu nét và $C_{k-1}$ là tranh trước đó. Nét vẽ sau nằm trên nét vẽ trước.

**Rasterizer (procedural), bộ vẽ thủ tục.** Đoạn mã vẽ một nét từ dãy số ra pixel bằng các phép tính hình học cố định:
dán texture, tô đa giác (`core/procedural_rasterizer.py`). Đây là bản chép *chính xác từng pixel* của mã gốc, dùng để
vẽ tranh cuối cùng ở $512 \times 512$.

**Rasterization, raster hoá.** Việc biến mô tả hình học (toạ độ, kích thước) thành pixel.

---

## 4. Mạng neural

**Differentiable, khả vi.** Tính được đạo hàm: biết khi đổi một con số một chút thì kết quả đổi theo hướng nào, nhiều
hay ít. Rasterizer **không** khả vi vì nó làm tròn về pixel, tô đa giác và chọn texture theo ngưỡng; kết quả của nó
thay đổi theo kiểu bậc thang.

**Neural renderer, bộ vẽ neural.** Mạng neural nhận dãy tham số nét và vẽ ra ảnh của nét đó: màu $\hat F$ và độ che
$\hat A$. Nó khả vi, nên dùng được để tính gradient (`models/neural_renderer.py`).

**Surrogate, mô hình thay thế.** Vai trò của neural renderer: bắt chước rasterizer đủ giống để thay nó *trong lúc tối
ưu*. Tranh cuối vẫn do rasterizer thật vẽ.

**FusionNet.** Kiến trúc của neural renderer, gồm hai nhánh:

- nhánh *hình dạng* chỉ nhận tham số hình dạng và vẽ ra mặt nạ $m$;
- nhánh *màu* nhận toàn bộ tham số và vẽ ra trường màu $c$;
- ghép lại: $\hat F = c \odot m$, $\hat A = \alpha \cdot m$.

**Light / full renderer, bản nhẹ / bản đầy đủ.** Bản nhẹ vẽ ô $32 \times 32$, bản đầy đủ vẽ $128 \times 128$. Pipeline
dùng bản nhẹ đã huấn luyện sẵn của bài gốc (`original-light`).

**Training (renderer), huấn luyện bộ vẽ.** Sinh ngẫu nhiên tham số nét, cho rasterizer vẽ làm "đáp án", rồi dạy mạng
vẽ giống đáp án. Hàm mất mát là sai số bình phương (MSE).

**Epoch, lượt học.** Một vòng huấn luyện qua 50 000 nét mới. Mạng sơn dầu mà dự án tự huấn luyện lại (để kiểm chứng) học
100 lượt.

**Fidelity, độ trung thực.** Mức giống nhau giữa nét mạng neural vẽ và nét rasterizer vẽ, đo bằng PSNR trên các nét
kiểm tra cố định. Ví dụ bộ vẽ gốc sơn dầu bản nhẹ đạt 24.38 dB.

**Dilate / erode, nở / co.** Hai phép xử lý mép nét, kích thước $3 \times 3$. Màu nét được *nở* ra một pixel và độ che
được *co* lại một pixel, nên mép nét không bị viền tối. Bước này đáng giá $+2.71$ dB.

---

## 5. Tối ưu

**Loss function ($\mathcal{L}$), hàm mất mát.** Con số đo tranh sai so với ảnh bao nhiêu, càng nhỏ càng tốt. Khi vẽ,
dự án dùng **L1**: trung bình của $\lvert \text{tranh} - \text{ảnh} \rvert$ trên mọi pixel.

**Gradient, đạo hàm theo nhiều biến.** Với mỗi tham số, $\partial \mathcal{L} / \partial \theta$ cho biết tăng tham
số đó lên thì loss tăng hay giảm, và nhanh cỡ nào.

**Backpropagation / autograd, lan truyền ngược.** Cách PyTorch tự tính gradient của loss theo mọi tham số, đi ngược từ
loss qua phép ghép nét, qua neural renderer, về từng con số của từng nét.

**Gradient descent, hạ gradient.** Nhích mỗi tham số một chút theo hướng làm loss giảm:
$\theta \leftarrow \theta - \eta \, \nabla_\theta \mathcal{L}$. Giống người bịt mắt dò dốc để xuống đồi.

**Optimizer (RMSprop), bộ tối ưu.** Biến thể của gradient descent: chia bước nhảy cho độ lớn trung bình gần đây của
gradient, nên tham số nào cũng tiến với tốc độ hợp lý.

**Learning rate ($\eta$), tốc độ học.** Độ dài mỗi bước nhích. Pipeline dùng 0.004; bài gốc dùng 0.002.

**Cosine schedule, lịch cosin.** Learning rate bắt đầu ở mức cao rồi giảm dần theo đường cosin về gần 0, và bắt đầu lại
mỗi khi thêm một nét mới: bước đầu đi nhanh, cuối chỉnh tinh.

**Iteration / step, bước lặp.** Một lần "vẽ, đo loss, tính gradient, nhích tham số". Ở chế độ "Best quality", mỗi nét
mới có 55 bước, áp lên mọi nét đã có trong ô.

**Clamp, kẹp giá trị.** Sau mỗi bước, tham số bị kẹp lại vào khoảng cho phép: hình dạng trong $[0.1, 0.9]$ để nét không
dính vào mép ô, màu và độ che trong $[0, 1]$.

**Error map, bản đồ sai lệch.** Ảnh $\sum_{c} \lvert \text{ảnh} - \text{tranh} \rvert$: chỗ nào sáng là chỗ tranh còn
sai nhiều (`pipeline/stroke_sampler.py`).

**Sampling, bốc mẫu.** Chọn tâm của nét mới bằng cách bốc ngẫu nhiên theo bản đồ sai lệch, sau khi đã làm mờ (cửa sổ
$4 \times 4$) và lấy mũ 4, nên chỗ sai nhất gần như luôn được chọn. Màu ban đầu của nét lấy từ ảnh tại điểm đó. Cách này
tốt hơn đặt nét ngẫu nhiên đều $+0.87$ dB.

**Stroke allocation, chia nét.** Mỗi ô được bao nhiêu nét. *Equal* (bài gốc): mỗi ô 9 nét. *Error* (chế độ "Fast"): ô
còn sai nhiều được nhiều nét hơn, chia theo phương pháp D'Hondt, tối đa gấp đôi mức trung bình.

**Optimal transport / Sinkhorn loss, vận chuyển tối ưu.** Một loss bổ sung đo "phải dời bao nhiêu màu" để tranh khớp
ảnh. Dự án đã thử nhưng không thấy tác dụng: gradient của nó chỉ bằng khoảng 0.3 % gradient của L1. Vì vậy pipeline tắt
loss này ($\beta_{OT} = 0$).

---

## 6. Đánh giá

**PSNR (Peak Signal-to-Noise Ratio).** Đo tranh giống ảnh tới mức nào theo từng pixel:
$\text{PSNR} = 10 \log_{10}(1 / \text{MSE})$, đơn vị dB, càng cao càng giống. Tranh 500 nét thường đạt khoảng 20 dB.

**SSIM (Structural Similarity).** Đo độ giống về cấu trúc (độ sáng, độ tương phản, đường nét) trên từng vùng nhỏ. 1 là
giống hệt.

**LPIPS.** Đo độ khác biệt "theo cảm nhận của người", dựa trên đặc trưng của mạng AlexNet. Càng **thấp** càng giống.

**Seed, hạt giống ngẫu nhiên.** Con số cố định cho bộ sinh số ngẫu nhiên, để chạy lại thì ra đúng kết quả cũ. Các con số
chính trong báo cáo là trung bình của 3 seed.

**Ablation, thí nghiệm bỏ bớt.** Tắt hoặc thay *một* thành phần, giữ nguyên mọi thứ khác, rồi đo chênh lệch, để biết
thành phần đó đóng góp bao nhiêu. Ví dụ: bản đồ sai lệch so với đặt nét ngẫu nhiên, hay progressive so với fixed.

---

## 7. Đầu ra

**Stroke file (`strokes.npz`), file nét.** Lưu bảng $495 \times 12$ tham số nét. Từ file này có thể vẽ lại tranh hoặc
làm video bất cứ lúc nào.

**Time-lapse.** Video quá trình vẽ từng nét (MP4 H.264 và GIF), được vẽ lại từ file nét sau khi tranh xong, mất khoảng
2 giây cho 500 nét (`export/video_recorder.py`).

**Web app (Gradio).** Giao diện web để tải ảnh lên, chọn loại cọ, chế độ vẽ, tốc độ, rồi xem tranh và time-lapse. Có
tiếng Việt và tiếng Anh (`app/gradio_app.py`).

**Device, thiết bị tính.** `cuda` (GPU NVIDIA), `mps` (chip Apple) hoặc `cpu`. Trên RTX 3070, một tranh progressive mất
khoảng 12 giây; trên CPU khoảng 130 giây.

**Speed setting, chế độ tốc độ.** *Best quality*: mỗi ô 9 nét, mỗi nét 55 bước. *Fast*: chia nét theo sai lệch và mỗi
nét chỉ 11 bước, nhanh gấp hơn hai lần (5.2 s so với 12.1 s) mà PSNR gần như không đổi ($-0.01$ dB).
