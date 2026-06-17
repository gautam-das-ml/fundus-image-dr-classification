import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from torch.utils.data import DataLoader

from dataset import FundusDataset
from model import get_model


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
    model.load_state_dict(torch.load("models/dr_model.pth"))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            probs = torch.sigmoid(outputs)

            preds = (probs > 0.5).int()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs).flatten()
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    # -------------------------
    # Confusion Matrix
    # -------------------------

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("results/confusion_matrix.png")
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

    plt.savefig("results/roc_curve.png")
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

    plt.savefig("results/pr_curve.png")
    plt.show()
    plt.close()

    # -------------------------
    # Training Loss Curve
    # -------------------------

    loss_values = [
        0.1486,0.0815,0.0541,0.0412,0.0298,
        0.0241,0.0236,0.0187,0.0271,0.0121,
        0.0171,0.0172,0.0121,0.0059,0.0055,
        0.0119,0.0064,0.0062,0.0091,0.0096
    ]

    epochs = range(1, len(loss_values)+1)

    plt.figure()

    plt.plot(epochs, loss_values, marker='o')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")

    plt.savefig("results/training_loss.png")
    plt.show()
    plt.close()

    print("All plots displayed and saved in the results folder.")


if __name__ == "__main__":
    main()