import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
class FundusDataset(Dataset):

    def __init__(self, csv_file, image_dir):

        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir

        # convert labels to binary
        self.df["label"] = self.df["diagnosis"].apply(lambda x: 0 if x == 0 else 1)

        # image preprocessing
        self.transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor()
])


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):

        image_id = self.df.iloc[idx]["id_code"]
        label = self.df.iloc[idx]["label"]

        img_path = os.path.join(self.image_dir, image_id + ".png")

        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)

        return image, label