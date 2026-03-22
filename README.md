# Hướng dẫn Train Model Phân Loại Bệnh Lá Lúa

## 1. Cài đặt môi trường

```bash
py -3.12 -m venv env
source env/Scripts/activate      # Git Bash
# hoặc: env\Scripts\activate     # CMD/PowerShell
```

## 2. Cài thư viện

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> Không có GPU NVIDIA thì thay bằng: `pip install torch torchvision`

## 3. Chuẩn bị dataset

Tải folder `data/final/` từ Google Drive về đặt vào thư mục `data/`.

Cấu trúc dataset:

```
data/final/
├── train/
│   ├── bacterial_leaf_blight/
│   ├── brown_spot/
│   └── leaf_blast/
├── val/
│   ├── bacterial_leaf_blight/
│   ├── brown_spot/
│   └── leaf_blast/
└── test/
    ├── bacterial_leaf_blight/
    ├── brown_spot/
    └── leaf_blast/
```

## 4. Train model

```bash
python src/train.py --data data/final --batch 8 --name v3
```

Các tham số tùy chỉnh:

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--data` | `data/final` | Đường dẫn dataset |
| `--model` | `yolo11s-cls.pt` | Model YOLO (`yolo11n-cls.pt` / `yolo11s-cls.pt` / `yolo11m-cls.pt`) |
| `--imgsz` | `320` | Kích thước ảnh |
| `--batch` | `8` | Batch size (giảm về 4 nếu bị lỗi OOM) |
| `--epochs` | `150` | Số epochs tối đa |
| `--name` | `v2` | Tên lần chạy |
| `--resume` | | Tiếp tục train từ checkpoint, ví dụ `--resume runs/classify/v3/weights/last.pt` |

## 5. Đánh giá model

```bash
python src/evaluate.py eval --model runs/classify/v3/weights/best.pt --data data/final --split val
```

## 6. Dự đoán

Phân loại 1 ảnh:

```bash
python src/predict.py --model runs/classify/v3/weights/best.pt --source path/to/anh_la_lua.jpg
```

Phân loại cả thư mục:

```bash
python src/predict.py --model runs/classify/v3/weights/best.pt --source path/to/folder/
```
