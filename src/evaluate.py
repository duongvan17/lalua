"""
Đánh giá model YOLO phân loại bệnh lá lúa.
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def evaluate_model(model_path: str, data_dir: str, split: str = "test",
                   imgsz: int = 224):
    """Đánh giá model trên tập test/val."""
    print("=" * 60)
    print(f"ĐÁNH GIÁ MODEL: {model_path}")
    print(f"Split: {split}")
    print("=" * 60)

    model = YOLO(model_path)
    metrics = model.val(
        data=data_dir,
        split=split,
        imgsz=imgsz,
        plots=True,
    )

    # In kết quả
    print(f"\n--- KẾT QUẢ ---")
    print(f"Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"Top-5 Accuracy: {metrics.top5:.4f}")

    # Lưu kết quả
    results = {
        "model": model_path,
        "split": split,
        "top1_accuracy": float(metrics.top1),
        "top5_accuracy": float(metrics.top5),
    }

    output_dir = Path(model_path).parent.parent
    results_file = output_dir / "eval_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nKết quả đã lưu tại: {results_file}")

    return metrics


def compare_models(model_paths: list, data_dir: str, split: str = "test",
                   imgsz: int = 224):
    """So sánh nhiều models."""
    print("=" * 60)
    print("SO SÁNH MODELS")
    print("=" * 60)

    results = []
    for path in model_paths:
        print(f"\nĐang đánh giá: {path}")
        model = YOLO(path)
        metrics = model.val(data=data_dir, split=split, imgsz=imgsz, plots=False)

        results.append({
            "model": path,
            "top1": float(metrics.top1),
            "top5": float(metrics.top5),
        })

    # In bảng so sánh
    print(f"\n{'Model':<40} {'Top-1':>8} {'Top-5':>8}")
    print("-" * 60)
    for r in results:
        name = Path(r['model']).parent.parent.name
        print(f"{name:<40} {r['top1']:>8.4f} {r['top5']:>8.4f}")

    best = max(results, key=lambda x: x['top1'])
    print(f"\nModel tốt nhất (Top-1): {best['model']} ({best['top1']:.4f})")

    return results


def main():
    parser = argparse.ArgumentParser(description="Đánh giá model YOLO phân loại bệnh lá lúa")
    subparsers = parser.add_subparsers(dest='command')

    ev = subparsers.add_parser('eval', help='Đánh giá 1 model')
    ev.add_argument('--model', required=True, help='Path to model weights (.pt)')
    ev.add_argument('--data', default='data/raw/RiceLeafsDisease')
    ev.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ev.add_argument('--imgsz', type=int, default=224)

    cmp = subparsers.add_parser('compare', help='So sánh nhiều models')
    cmp.add_argument('--models', nargs='+', required=True)
    cmp.add_argument('--data', default='data/raw/RiceLeafsDisease')
    cmp.add_argument('--split', default='test')
    cmp.add_argument('--imgsz', type=int, default=224)

    args = parser.parse_args()

    if args.command == 'eval':
        evaluate_model(args.model, args.data, args.split, args.imgsz)
    elif args.command == 'compare':
        compare_models(args.models, args.data, args.split, args.imgsz)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
