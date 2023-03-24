import os
from torch.utils.data import Dataset
import csv
from PIL import Image

ROOT_PATH = os.environ["PREFGEN_ROOT"]

class FECNetDataset(Dataset):
    train_labels = os.path.join(
        ROOT_PATH,
        "prefgen/data/FECDataset/train_labels.csv"
    )
    test_labels = os.path.join(
        ROOT_PATH,
        "prefgen/data/FECDataset/test_labels.csv"
    )
    data_path = os.path.join(
        ROOT_PATH,
        "prefgen/data/FECDataset"
    )
    
    def __init__(self, split="train", transform=None):
        super().__init__()
        self.transform = transform
        self.split = split
        # Load the labels
        if split == "train":
            labels_path = self.train_labels
        elif split == "test":
            labels_path = self.test_labels

        with open(labels_path, "r") as f:
            # Load csv into list
            self.labels = list(csv.reader(f))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label_row = self.labels[idx]
        label = label_row[-1] # Here the label is the negative image
        image_one_path = os.path.join(
            self.data_path,
            label_row[1]
        )
        image_two_path = os.path.join(
            self.data_path,
            label_row[2]
        )
        image_three_path = os.path.join(
            self.data_path,
            label_row[3]
        )
        image_one = Image.open(image_one_path)
        image_two = Image.open(image_two_path)
        image_three = Image.open(image_three_path)

        if label == "1":
            return image_two, image_three, image_one
        elif label == "2":
            return image_one, image_three, image_two
        elif label == "3":
            return image_one, image_two, image_three
