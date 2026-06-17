import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from torch.utils.data import DataLoader

from dataset import FundusDataset
from model_2 import get_model


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FundusDataset(
        csv_file="data/test/test.csv",
        image_dir="data/test/images"
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )

    model = get_model()
    model.load_state_dict(torch.load("models/efficientnet_b0.pth"))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)

            all_probs.extend(probs[:,1].cpu().numpy().flatten())

            all_preds.extend(preds.cpu().numpy().flatten())

            all_labels.extend(labels.cpu().numpy().flatten())

    # -------------------------
    # Confusion Matrix
    # -------------------------

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("results/confusion_matrix_effnet.png")
    plt.show()
    plt.close()

    # -------------------------
    # ROC Curve
    # -------------------------

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()

    plt.savefig("results/roc_curve_effnet.png")
    plt.show()
    plt.close()

    # -------------------------
    # Precision-Recall Curve
    # -------------------------

    precision, recall, _ = precision_recall_curve(all_labels, all_probs)

    plt.figure()

    plt.plot(recall, precision)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    plt.savefig("results/pr_curve_effnet.png")
    plt.show()
    plt.close()

    # -------------------------
    # Training Loss Curve
    # -------------------------

    loss_values = [
        0.2334,0.0891,0.0552,0.0445,0.0452,
        0.0284,0.0254,0.0227,0.0124,0.0142,
        0.0126,0.0095,0.0109,0.0131,0.0092,
        0.0054,0.0046,0.0058, 0.0078,0.0084,
        0.0070,0.0114,0.0075,0.0057,0.0037
    ]

    epochs = range(1, len(loss_values)+1)

    plt.figure()

    plt.plot(epochs, loss_values, marker='o')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")

    plt.savefig("results/training_loss_effnet.png")
    plt.show()
    plt.close()

    print("All plots displayed and saved in the results folder.")


if __name__ == "__main__":
    main()