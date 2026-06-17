import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FundusDataset
from model import get_model


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = FundusDataset(
        csv_file="data/train/train.csv",
        image_dir="data/train/images"
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=6,
        pin_memory=True
    )

    model = get_model().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    epochs = 20

    for epoch in range(epochs):

        total_loss = 0

        for images, labels in loader:

            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "models/dr_model.pth")


if __name__ == "__main__":
    main()