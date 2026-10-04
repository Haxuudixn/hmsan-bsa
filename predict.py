"""Command-line inference for HMSAN-BSA.

Example:
    python predict.py --checkpoint outputs/hmsan_bsa_ft2/best_model.pt \
        --data_dir "C:\\Users\\Administrator\\Desktop\\新建文件夹 (2)" \
        --output predictions.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from bid_slicing.inference import HMSANPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HMSAN-BSA inference")
    parser.add_argument("--checkpoint", required=True, help="Trained model checkpoint")
    parser.add_argument("--data_dir", default=None, help="Directory with training_data_*.zip files")
    parser.add_argument("--zips", nargs="*", default=None, help="Explicit ZIP files")
    parser.add_argument("--label_config", default="configs/labels.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="predictions.json")
    parser.add_argument("--ocr_gpu", action="store_true", help="Run PaddleOCR on GPU")
    args = parser.parse_args()

    if args.data_dir:
        zips = sorted(Path(args.data_dir).glob("training_data_*.zip"))
    else:
        zips = [Path(z) for z in (args.zips or [])]
    if not zips:
        print("No input ZIPs found. Pass --data_dir or --zips.")
        sys.exit(1)

    predictor = HMSANPredictor(
        checkpoint_path=args.checkpoint,
        label_config=args.label_config,
        device=args.device,
    )
    documents_loaded = []
    from bid_slicing.data.dataset import load_documents_from_zips
    documents_loaded = load_documents_from_zips([str(z) for z in zips])

    print(f"Loaded {len(documents_loaded)} documents from {len(zips)} ZIP(s).")
    print("Running PaddleOCR on missing image OCR text, then inference...")
    results = predictor.predict_to_json(
        documents_loaded,
        output_path=args.output,
        use_gpu=args.ocr_gpu,
    )
    print(f"Predictions saved to {args.output}")
    print(f"Documents predicted: {len(results)}")


if __name__ == "__main__":
    main()
