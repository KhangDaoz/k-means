# K-Means Player Clustering

Dự án này đọc dữ liệu từ `data.csv`, tiền xử lý các cột số, chuẩn hóa đặc trưng và chạy K-Means để phân cụm người chơi. Script cũng tạo các ảnh minh họa cho từng bước lặp và kết quả cuối cùng.

## Yêu cầu

- Python 3.10+
- Các thư viện: `pandas`, `numpy`, `matplotlib`, `scipy`, `scikit-learn`

## Cài đặt

```bash
pip install pandas numpy matplotlib scipy scikit-learn
```

## Chạy chương trình

```bash
python main.py
```

## Kết quả

Khi chạy xong, chương trình sẽ tạo các file ảnh như:

- `kmeans_step_initial.png`
- `kmeans_step_0_init_centroids.png`
- `kmeans_step_00.png`, `kmeans_step_01.png`, ...
- `kmeans_step_final.png`

## Ghi chú

- Dữ liệu đầu vào phải nằm cùng thư mục với `main.py` và có tên `data.csv`.
- Mặc định chương trình chạy với `k=3` và `max_iter=50`.
