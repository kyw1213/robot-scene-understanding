import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MITIndoor67Dataset
from models import build_model


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def main():
    cfg = load_config("configs/resnet18.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(cfg["output_dir"])
    pred_dir = output_dir / "predictions"
    fig_dir = output_dir / "figures"
    pred_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    dataset = MITIndoor67Dataset(
        cfg["dataset_root"],
        cfg["test_list"],
        cfg["image_size"],
        train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    ckpt_path = output_dir / "checkpoints" / "best_resnet18.pth"
    checkpoint = torch.load(ckpt_path, map_location=device)

    model = build_model(
        cfg["model"],
        cfg["num_classes"],
        cfg["pretrained"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    all_probs = []

    total_loss = 0.0
    total = 0

    for images, labels in tqdm(loader, desc="eval"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        total += labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.max(dim=1).values.cpu().tolist())

    acc = sum(p == y for p, y in zip(all_preds, all_labels)) / len(all_labels)
    avg_loss = total_loss / total

    print(f"Test loss: {avg_loss:.4f}")
    print(f"Test accuracy: {acc:.4f}")

    report = classification_report(
        all_labels,
        all_preds,
        target_names=dataset.classes,
        digits=4,
        zero_division=0,
    )
    print(report)

    with open(pred_dir / "classification_report.txt", "w") as f:
        f.write(report)

    results = []
    for idx, (pred, label, conf) in enumerate(zip(all_preds, all_labels, all_probs)):
        image_path, _ = dataset.samples[idx]
        results.append({
            "image_path": str(image_path),
            "label": dataset.classes[label],
            "prediction": dataset.classes[pred],
            "confidence": float(conf),
            "correct": pred == label,
        })

    with open(pred_dir / "predictions_resnet18.json", "w") as f:
        json.dump(results, f, indent=2)

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(18, 16))
    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=dataset.classes,
        yticklabels=dataset.classes,
        cbar=True,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("MIT Indoor 67 Confusion Matrix")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix_resnet18.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()

