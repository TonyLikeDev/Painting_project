# Proposal corrections (Đề cương đồ án cơ sở)

Source document: `Do an co so Nguyen Le Hoang.pdf` (6 pages). These edits align the proposal with the decisions table in `RESEARCH_PLAN.md`, Section 0. Apply them to the Word/PDF source before submitting the final proposal (Week 1 exit criterion: "proposal submitted").

---

## (a) Replacement text for Section 4.1 "Phần mềm" (Vietnamese)

Replace the whole of Section 4.1 with:

```
4.1. Phần mềm
• Ứng dụng Python (Python 3.11 trở lên, PyTorch 2.x):
  o Giao diện web xây dựng bằng Gradio, chạy trên trình duyệt
  o Hỗ trợ chạy trên GPU NVIDIA (CUDA), Apple Silicon (MPS) và CPU
• Có các chức năng:
  o Tiếp nhận ảnh đầu vào (tải lên JPG/PNG hoặc chọn ảnh mẫu có sẵn)
  o Chọn chất liệu cọ (sơn dầu, màu nước, bút dạ, băng dính màu) và cấu hình tham số vẽ
    (số lượng nét cọ, chế độ chia lưới, màu nền)
  o Tối ưu hóa nét cọ với thanh tiến trình và xem trước kết quả theo thời gian thực
  o Hiển thị so sánh ảnh gốc và tranh hoàn thiện; xuất kết quả (ảnh PNG/JPG, video MP4/GIF)
```

## (b) English translation of the replacement

```
4.1. Software
• Python application (Python 3.11 or later, PyTorch 2.x):
  o Web interface built with Gradio, running in the browser
  o Runs on NVIDIA GPU (CUDA), Apple Silicon (MPS) and CPU
• Functions:
  o Image input (upload JPG/PNG or pick a bundled sample image)
  o Brush material selection (oil paint, watercolor, marker pen, colored tape) and painting
    parameters (stroke count, grid mode, canvas color)
  o Stroke optimization with a progress bar and a live preview of the result
  o Side-by-side display of the input photo and the finished painting; export of results
    (PNG/JPG image, MP4/GIF video)
```

## (c) Other places that conflict with the decisions in RESEARCH_PLAN.md Section 0

| # | Location in proposal | Current text | Suggested one-line edit (Vietnamese) | Reason |
| :-- | :--- | :--- | :--- | :--- |
| 1 | Section 2.2, group "Phát triển giao diện và thử nghiệm", first bullet | "Xây dựng giao diện ứng dụng (Web App / Desktop GUI) thân thiện, cho phép người dùng tùy chỉnh phong cách cọ vẽ, mật độ nét cọ và theo dõi kết quả." | "Xây dựng giao diện ứng dụng web (Gradio) thân thiện, cho phép người dùng tùy chỉnh chất liệu cọ, mật độ nét cọ, chế độ chia lưới và theo dõi kết quả." | Decision: Gradio web UI, not desktop GUI. |
| 2 | Section 2.2, group "Thuật toán và mô hình hóa", second bullet | "...các bộ chất liệu cọ vẽ đa dạng như sơn dầu (oil-paint), màu nước (watercolor) và bút dạ (marker pen)." | "...các bộ chất liệu cọ vẽ đa dạng như sơn dầu (oil-paint), màu nước (watercolor), bút dạ (marker pen) và băng dính màu (colored tape)." | Section 3.2 and Week 6 list four materials; Section 2.2 lists only three. |
| 3 | Section 5, Tuần 1, third bullet | "Lựa chọn công nghệ, framework (PyTorch/TensorFlow) và thiết lập môi trường phát triển" | "Sử dụng Python 3.11+ và PyTorch 2.x; thiết lập môi trường phát triển trên GPU NVIDIA (CUDA), Apple Silicon (MPS) và CPU" | Decision: PyTorch 2.x with mps / cuda / cpu back ends. |
| 4 | Section 5, Tuần 4, first bullet | "Cài đặt/tinh chỉnh Differentiable Neural Renderer cho một chất liệu cọ (sơn dầu)" | "Cài đặt Differentiable Neural Renderer cho chất liệu sơn dầu: dùng renderer tiền huấn luyện của công trình gốc làm cơ sở so sánh, sau đó huấn luyện renderer riêng trên dữ liệu nét cọ tổng hợp" | Decision: load pretrained checkpoints first, then train our own renderer. |
| 5 | Section 5, Tuần 8, first bullet | "Thiết kế và cài đặt giao diện Web App/Desktop GUI" | "Thiết kế và cài đặt giao diện web bằng Gradio" | Same as row 1. |
| 6 | Section 5, Tuần 9, third bullet | "Đánh giá hiệu năng hệ thống: chất lượng thẩm mỹ và thời gian xử lý trên nhiều cấu hình phần cứng" | "Đánh giá hiệu năng hệ thống: chất lượng (PSNR, SSIM, LPIPS và khảo sát thẩm mỹ), thời gian xử lý và bộ nhớ trên các cấu hình CUDA, Apple Silicon (MPS) và CPU" | Decision: evaluation and benchmarking are first-class work with named metrics. |

Places checked and found consistent (no edit needed):
- Section 3 (Chức năng hệ thống): all four sub-sections match the Gradio feature list.
- Section 4.2 (Báo cáo, 40–60 trang) and 4.3 (Demo, 5–10 phút): unchanged.
- Section 5, Tuần 3, bullet "Cài đặt bộ chia lưới (grid rendering)": already describes only the image splitter, which matches the decision that the progressive loop is built in Week 7. No edit needed.
- Section 5, Tuần 6: the four materials are listed correctly.

## (d) Section 5 week titles

The ten week titles in Section 5 ("Tuần 1: Nghiên cứu tổng quan và hoàn thiện đề cương" through "Tuần 10: Hoàn thiện báo cáo, demo và chuẩn bị bảo vệ") stay exactly as written. Only the bullet contents listed in (c) change. RESEARCH_PLAN.md Section 4 uses these titles as its master schedule.
