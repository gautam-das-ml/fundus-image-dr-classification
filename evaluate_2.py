import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from dataset import FundusDataset
from model_2 import get_model

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load validation dataset
dataset = FundusDataset(
    csv_file="data/test/test.csv",
    image_dir="data/test/images"
)

loader = DataLoader(dataset, batch_size=32, shuffle=False)

# load trained model
model = get_model()
model.load_state_dict(torch.load("models/efficientnet_b0.pth"))
model = model.to(device)

model.eval()


all_preds = []
all_labels = []


with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs,1)
        

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())


# accuracy
acc = accuracy_score(all_labels, all_preds)

print("Accuracy:", acc)


# confusion matrix
cm = confusion_matrix(all_labels, all_preds)

print("Confusion Matrix:")
print(cm)