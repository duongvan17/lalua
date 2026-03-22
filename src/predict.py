"""
Inference pipeline cho model YOLO phân loại bệnh lá lúa.
Đưa ảnh vào → trả về tên bệnh + độ tự tin.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


# Tên tiếng Việt cho các loại bệnh
DISEASE_VI = {
    "bacterial_leaf_blight": "Bạc lá",
    "brown_spot": "Đốm nâu",
    "healthy": "Khỏe mạnh",
    "leaf_blast": "Đạo ôn",
    "leaf_scald": "Cháy bìa lá",
    "narrow_brown_spot": "Đốm nâu hẹp",
}


def predict(model_path: str, source: str, imgsz: int = 320, conf: float = 0.25,
            save: bool = True, show: bool = False):
    """Phân loại ảnh lá lúa."""
    model = YOLO(model_path)

    results = model.predict(
        source=source,
        imgsz=imgsz,
        save=save,
        show=show,
    )

    for r in results:
        probs = r.probs
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()
        class_name = model.names[top1_idx]
        vi_name = DISEASE_VI.get(class_name, class_name)

        print(f"\n--- {Path(r.path).name} ---")
        print(f"  Kết quả: {class_name} ({vi_name})")
        print(f"  Độ tự tin: {top1_conf:.1%}")

        # Top 5 predictions
        top5_idx = probs.top5
        top5_conf = probs.top5conf.tolist()
        print(f"  Top 5:")
        for idx, c in zip(top5_idx, top5_conf):
            name = model.names[idx]
            vi = DISEASE_VI.get(name, name)
            print(f"    {name} ({vi}): {c:.1%}")

    return results


def predict_single(model_path: str, image_path: str, imgsz: int = 320):
    """Predict 1 ảnh, trả về dict."""
    model = YOLO(model_path)
    results = model.predict(source=image_path, imgsz=imgsz, verbose=False)

    r = results[0]
    probs = r.probs
    top1_idx = probs.top1
    top1_conf = probs.top1conf.item()
    class_name = model.names[top1_idx]

    return {
        'class': class_name,
        'class_vi': DISEASE_VI.get(class_name, class_name),
        'confidence': top1_conf,
        'all_probs': {model.names[i]: probs.data[i].item() for i in range(len(model.names))},
    }


def main():
    parser = argparse.ArgumentParser(description="Phân loại bệnh lá lúa")
    parser.add_argument('--model', required=True, help='Path to model weights (.pt)')
    parser.add_argument('--source', required=True, help='Ảnh hoặc thư mục ảnh')
    parser.add_argument('--imgsz', type=int, default=320)
    parser.add_argument('--show', action='store_true', help='Hiển thị ảnh')

    args = parser.parse_args()

    predict(
        model_path=args.model,
        source=args.source,
        imgsz=args.imgsz,
        show=args.show,
    )


if __name__ == "__main__":
    main()
