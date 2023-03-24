"""
    Code for training a triplet network on the
    FECNet dataset.
"""
import argparse
from lib2to3.pytree import NegatedPattern
from pickletools import optimize
from prefgen.methods.attribute_classifiers.face_identification.w_space_to_identity import DenseEmbedder
from prefgen.methods.attribute_classifiers.facial_expression.densenet import DenseNet
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm
import wandb
from PIL import Image
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
import cv2
# from pytorch_metric_learning import miners, losses

############################### FECNet Data Loader ###############################

class FECNetDataset(Dataset):
    """Google facial expression dataset."""
    data_paths = {
        "train": os.path.join(
            os.environ["PREFGEN_ROOT"], 
            "prefgen/data/FECDataset", 
            "train_labels.csv"
        ), 
        "test": os.path.join(
            os.environ["PREFGEN_ROOT"], 
            "prefgen/data/FECDataset",
            "test_labels.csv"
        ),
    }

    image_root_path = os.path.join(
        os.environ["PREFGEN_ROOT"],
        "prefgen/data/FECDataset"
    )

    def __init__(self, split="train", transform=None, get_paths=False):
        """
        Args:
            split (string, optional):  Optional string saying which data split
                the dataset should come from. 
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert split in ["train", "test"]
        self.split = split
        self.triplet_df = pd.read_csv(
            FECNetDataset.data_paths[self.split],
            header=None,
            # error_bad_lines=False
            on_bad_lines="skip"
        )
        self.get_paths = get_paths

    def __len__(self):
        return len(self.triplet_df)

    def _make_image_path(self, file_name):
        """
            Returns the local path to an image given the url
        """
        # file_name = list(url.split("/"))[-1]
        image_path = os.path.join(
            FECNetDataset.image_root_path,
            file_name
        )
        return image_path

    def __getitem__(self, idx):
        """
            Returns a (3, colors, width, height) image triplet
            from `idx` line in the dataframe.  
        """
        # Load the paths
        image_1_path = self._make_image_path(self.triplet_df.loc[idx, 1])
        image_2_path = self._make_image_path(self.triplet_df.loc[idx, 2])
        image_3_path = self._make_image_path(self.triplet_df.loc[idx, 3])
        # Mode determines the triplet order
        mode = self.triplet_df.iloc[idx, -1]
        if self.get_paths:
            if mode == 1: 
                triplet = (image_3_path, image_2_path, image_1_path)
            elif mode == 2:
                triplet = (image_1_path, image_3_path, image_2_path)
            elif mode == 3:
                triplet = (image_1_path, image_2_path, image_3_path)
            else:
                return self.__getitem__((idx - 1) % self.__len__())
            return triplet

        # Load the three images
        try:
            image_1 = cv2.imread(image_1_path)
            image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
            image_1 = torch.Tensor(image_1) / 255
            image_1 = torch.permute(image_1, (2, 0, 1))
            image_2 = cv2.imread(image_2_path)
            image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
            image_2 = torch.Tensor(image_2) / 255
            image_2 = torch.permute(image_2, (2, 0, 1))
            image_3 = cv2.imread(image_3_path)
            image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2RGB)
            image_3 = torch.Tensor(image_3) / 255
            image_3 = torch.permute(image_3, (2, 0, 1))
            """
            image_1 = transforms.ToTensor()(Image.open(image_1_path))
            image_2 = transforms.ToTensor()(Image.open(image_2_path))
            image_3 = transforms.ToTensor()(Image.open(image_3_path))
            """
        except:
            return self.__getitem__((idx - 1) % self.__len__())
        if mode == 1: 
            triplet = torch.stack((image_3, image_2, image_1))
        elif mode == 2:
            triplet = torch.stack((image_1, image_3, image_2))
        elif mode == 3:
            triplet = torch.stack((image_1, image_2, image_3))
        else:
            raise Exception(f"Unrecognized mode index {mode}")
        # Return data
        return triplet

def load_fecnet_dataloaders(batch_size=30, num_workers=4):
    """
        Load dataloaders 
    """
    # Load the datasets
    train_dataset = FECNetDataset(split="train")
    test_dataset = FECNetDataset(split="test")
    # Make the dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        drop_last=True
    )

    return train_dataloader, test_dataloader

############################## Evaluation Metrics #########################

def percentage_of_triplets_satisfied(anchors, positives, negatives):
    """
        Measures percentage of given triplets that are
        satisfied based on the given embeddings. 
    """
    is_satisfied = []
    for index in range(len(anchors)):
        anchor = anchors[index]
        positive = positives[index]
        negative = negatives[index]
        satisfied = torch.norm(anchor - positive) < torch.norm(anchor - negative).item()
        is_satisfied.append(satisfied)

    return sum(is_satisfied) / len(is_satisfied)

############################## Training Code ##############################

def test_model(model, batch_size=32, num_workers=4):
    _, test_dataloader = load_fecnet_dataloaders(
        batch_size=batch_size, 
        num_workers=num_workers
    )
    triplet_margin_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    # Set model to evaluation
    model.eval()
    triplet_losses_over_time = []
    percentage_satisfied_over_time = []
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(tqdm(test_dataloader, position=0, leave=True)):
            sample_batched = sample_batched.cuda()
            # Unflatten the 
            # Batch should be of size (batch_size, 3, num_colors, width, height)
            # Flatten in column major order so we can separate anchor, positive, negative
            sample_batched = sample_batched.transpose(0, 1)
            sample_batched = torch.flatten(sample_batched, start_dim=0, end_dim=1)
            # Forward pass
            embeddings = model(sample_batched)
            # Separate into batches of anchor, positive, negatives
            anchors = embeddings[0: batch_size]
            positives = embeddings[batch_size: 2 * batch_size]
            negatives = embeddings[2 * batch_size:]
            assert len(anchors) == len(positives) and len(positives) == len(negatives)
            # Measure the triplet loss
            triplet_loss = triplet_margin_loss(anchors, positives, negatives).item()
            triplet_losses_over_time.append(triplet_loss)
            percentage_satisfied = percentage_of_triplets_satisfied(anchors, positives, negatives)
            # Log metrics
            percentage_satisfied_over_time.append(percentage_satisfied.item())

    print(np.mean(percentage_satisfied_over_time))

def run_eval_loop(args, model, test_dataloader, num_batches=100):
    triplet_margin_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    # Set model to evaluation
    model.eval()
    triplet_losses_over_time = []
    percentage_satisfied_over_time = []
    with torch.no_grad():
        for j_batch, sample_batched in enumerate(tqdm(test_dataloader, position=0, leave=True)):
            if j_batch > num_batches: 
                break
            sample_batched = sample_batched.cuda()
            # Unflatten the 
            # Batch should be of size (batch_size, 3, num_colors, width, height)
            # Flatten in column major order so we can separate anchor, positive, negative
            sample_batched = sample_batched.transpose(0, 1)
            sample_batched = torch.flatten(sample_batched, start_dim=0, end_dim=1)
            # Forward pass
            embeddings = model(sample_batched)
            assert embeddings.shape[-1] == args.embedding_dim
            # Separate into batches of anchor, positive, negatives
            anchors = embeddings[0: args.batch_size]
            positives = embeddings[args.batch_size: 2 * args.batch_size]
            negatives = embeddings[2 * args.batch_size:]
            assert len(anchors) == len(positives) and len(positives) == len(negatives)
            # Measure the triplet loss
            triplet_loss = triplet_margin_loss(anchors, positives, negatives).item()
            triplet_losses_over_time.append(triplet_loss)
            percentage_satisfied = percentage_of_triplets_satisfied(anchors, positives, negatives)
            percentage_satisfied_over_time.append(percentage_satisfied.detach().cpu().numpy())
        # Log metrics
        wandb.log({
            "Test Triplet Loss": np.mean(triplet_losses_over_time),
            "Test Percentage Satisfied": np.mean(percentage_satisfied_over_time),
        })

    return np.mean(triplet_losses_over_time)

def train(model, args):
    """
        Training code 
    """
    print("Starting training")
    # Setup the Optimizer
    # optimizer = Adam(model.parameters(), lr=args.learning_rate)
    optimizer = SGD(model.parameters(), lr=args.learning_rate)
    # Setup the loss metrics
    triplet_margin_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    # Load Dataloaders
    train_dataloader, test_dataloader = load_fecnet_dataloaders(
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    # TODO Setup triplet mining
    if args.triplet_mining:
        triplet_miner_func = miners.TripletMarginMiner(
            margin=1.0,
            type_of_triplets="all"
        )

    best_triplet_loss = float("inf")
    # Go through the epochs
    for epoch in tqdm(range(args.epochs)):
        # Set model to train
        model.train()
        # Training Loop
        for i_batch, sample_batched in enumerate(tqdm(train_dataloader, position=0, leave=True)):
            optimizer.zero_grad()
            sample_batched = sample_batched.cuda()
            # Batch should be of size (batch_size, 3, num_colors, width, height)
            # Flatten in column major order so we can separate anchor, positive, negative
            sample_batched = sample_batched.transpose(0, 1)
            sample_batched = torch.flatten(sample_batched, start_dim=0, end_dim=1)
            # Forward pass
            embeddings = model(sample_batched)
            # Separate into batches of anchor, positive, negatives
            anchors = embeddings[0: args.batch_size]
            positives = embeddings[args.batch_size: 2 * args.batch_size]
            negatives = embeddings[2 * args.batch_size:]
            assert len(anchors) == len(positives) and len(positives) == len(negatives)
            # Compute loss metrics
            triplet_loss = triplet_margin_loss(anchors, positives, negatives)
            percentage_satisfied = percentage_of_triplets_satisfied(anchors, positives, negatives)
            # Optimizer step
            triplet_loss.backward()
            optimizer.step()
            # Log metrics
            wandb.log({
                "Train Triplet Loss": triplet_loss,
                "Train Percentage Satisfied": percentage_satisfied,
                "epoch": epoch
            })

        # Testing 
        mean_triplet_loss = run_eval_loop(args, model, test_dataloader, num_batches=100)
        if mean_triplet_loss < best_triplet_loss:
            torch.save(model.state_dict(), args.model_save_path)
            best_triplet_loss = mean_triplet_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch FECNet')
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--embedding_dim', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--triplet_mining', default=False)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--num_workers', default=4)
    parser.add_argument(
        '--model_save_path', 
        default=os.path.join(
            os.environ["PREFGEN_ROOT"],
            "prefgen/pretrained/face_expression/facenet_modified_best_{16}.pt"
        )
    )

    args = parser.parse_args()
    # Setup wandb
    run = wandb.init(
        project="GanSearch",
        group="FECNet Training",
        config=args
    )
    # Set up seeds
    torch.manual_seed(0)
    np.random.seed(0)
    # Load the model
    model = load_facenet_model(embedding_dim=args.embedding_dim).to(args.device)
    # Run training on the model
    train(model, args)
    