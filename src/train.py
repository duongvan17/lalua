"""
Training script cho YOLO11 phân loại bệnh lá lúa v2.
Tối ưu cho generalization trên nhiều nguồn ảnh + VRAM 4GB.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(data_dir: str, model_name: str = "yolo11s-cls.pt", epochs: int = 150,
          imgsz: int = 320, batch: int = 8, project: str = "runs/classify",
          name: str = "v2"):
    """Train model với augmentation mạnh + regularization."""
    print("=" * 60)
    print("TRAINING V2: YOLO11 Classification")
    print(f"Model: {model_name} | imgsz: {imgsz} | batch: {batch}")
    print(f"Data: {data_dir}")
    print("=" * 60)

    model = YOLO(model_name)
    results = model.train(
        data=data_dir,
        task="classify",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=30,

        # Optimizer - AdamW cho dataset nhỏ đa dạng
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.01,
        warmup_epochs=5.0,
        cos_lr=True,

        # Augmentation MẠNH - bridge domain gap
        degrees=30.0,
        translate=0.2,
        scale=0.7,
        shear=5.0,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        erasing=0.3,

        # Regularization
        dropout=0.3,

        # Other
        workers=4,
        seed=42,
        deterministic=True,
        val=True,
        plots=True,
        save_period=10,
        project=project,
        name=name,
    )

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining hoàn tất. Best model: {best_path}")
    return str(best_path)


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 phân loại bệnh lá lúa v2")
    parser.add_argument('--data', type=str, default='data/final',
                        help='Path to dataset (folder chứa train/ và val/)')
    parser.add_argument('--model', type=str, default='yolo11s-cls.pt',
                        help='YOLO model (yolo11n-cls/s-cls/m-cls.pt)')
    parser.add_argument('--imgsz', type=int, default=320, help='Image size')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Epochs')
    parser.add_argument('--resume', type=str, default=None, help='Resume từ checkpoint')
    parser.add_argument('--project', type=str, default='runs/classify', help='Project directory')
    parser.add_argument('--name', type=str, default='v2', help='Run name')

    args = parser.parse_args()

    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    train(
        data_dir=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
