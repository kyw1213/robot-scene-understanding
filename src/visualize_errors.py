import json
import shutil
from pathlib import Path


def main():
    pred_path = Path("outputs/predictions/predictions_resnet18.json")
    output_dir = Path("outputs/figures/error_cases")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(pred_path, "r") as f:
        results = json.load(f)

    errors = [item for item in results if not item["correct"]]
    errors = sorted(errors, key=lambda x: x["confidence"], reverse=True)

    max_cases = 50

    for idx, item in enumerate(errors[:max_cases]):
        image_path = Path(item["image_path"])
        label = item["label"]
        pred = item["prediction"]
        conf = item["confidence"]

        safe_name = f"{idx:03d}_gt-{label}_pred-{pred}_conf-{conf:.2f}{image_path.suffix}"
        target_path = output_dir / safe_name

        shutil.copy(image_path, target_path)

    print(f"Saved {min(len(errors), max_cases)} error cases to {output_dir}")


if __name__ == "__main__":
    main()

