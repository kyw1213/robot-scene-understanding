import json
from pathlib import Path

import open_clip
import torch
import yaml
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MITIndoor67Dataset


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_prompts(classes):
    prompts = []
    for name in classes:
        clean_name = name.replace("_", " ")
        prompts.append(f"a photo of a {clean_name}")
    return prompts


@torch.no_grad()
def main():
    cfg = load_config("configs/resnet18.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(cfg["output_dir"])
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    dataset = MITIndoor67Dataset(
        cfg["dataset_root"],
        cfg["test_list"],
        cfg["image_size"],
        train=False,
    )

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    model = model.to(device)
    model.eval()

    prompts = build_prompts(dataset.classes)
    text_tokens = tokenizer(prompts).to(device)

    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    all_preds = []
    all_labels = []
    results = []

    for image_path, label in tqdm(dataset.samples, desc="clip zero-shot"):
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = 100.0 * image_features @ text_features.T
        probs = logits.softmax(dim=-1)

        pred = probs.argmax(dim=-1).item()
        conf = probs[0, pred].item()

        all_preds.append(pred)
        all_labels.append(label)

        results.append({
            "image_path": str(image_path),
            "label": dataset.classes[label],
            "prediction": dataset.classes[pred],
            "confidence": float(conf),
            "correct": pred == label,
        })

    acc = sum(p == y for p, y in zip(all_preds, all_labels)) / len(all_labels)
    print(f"CLIP zero-shot accuracy: {acc:.4f}")

    report = classification_report(
        all_labels,
        all_preds,
        target_names=dataset.classes,
        digits=4,
        zero_division=0,
    )
    print(report)

    with open(pred_dir / "clip_zero_shot_predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(pred_dir / "clip_zero_shot_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    torch.save(
        {
            "confusion_matrix": cm,
            "classes": dataset.classes,
            "accuracy": acc,
            "prompts": prompts,
        },
        pred_dir / "clip_zero_shot_metrics.pt",
    )


if __name__ == "__main__":
    main()
