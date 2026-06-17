from dataset import FundusDataset
from torch.utils.data import DataLoader

dataset = FundusDataset(
    csv_file="data/train/train.csv",
    image_dir="data/train/images"
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)

print("Number of batches:", len(loader))

for images, labels in loader:
    print("Batch image shape:", images.shape)
    print("Batch labels shape:", labels.shape)
    break