"""
    Trains a classifier that maps w-space vectors to 
    identity space formed by the ArcFace model used in 
    IDLoss().
"""
from prefgen.methods.attribute_classifiers.face_identification.id_loss import IDLoss
from prefgen.methods.generative_models.stylegan2.stylegan_wrapper import StyleGAN2Wrapper
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import wandb
import pickle
import argparse
import torch.multiprocessing as mp

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass
############################### Attribute classifier model ###############################

def get_norm(n_filters, norm):
    if norm is None:
        return nn.Identity()
    elif norm == "batch":
        return nn.BatchNorm2d(n_filters, momentum=0.9)
    elif norm == "instance":
        return nn.InstanceNorm2d(n_filters, affine=True)
    elif norm == "layer":
        return nn.GroupNorm(1, n_filters)

class DenseEmbedder(nn.Module):
    """Supposed to map small-scale features (e.g. labels) to some given latent dim"""
    def __init__(self, input_dim, up_dim=128, depth=4, num_classes_list=[2], given_dims=None, norm=None):
        super().__init__()
        self.net = nn.ModuleList()
        if given_dims is not None:
            assert given_dims[0] == input_dim
            assert given_dims[-1] == up_dim
            dims = given_dims
        else:
            dims = np.linspace(input_dim, up_dim, depth).astype(int)

        for l in range(len(dims)-1):
            self.net.append(nn.Conv2d(dims[l], dims[l + 1], 1))

            self.net.append(get_norm(dims[l + 1], norm))
            self.net.append(nn.LeakyReLU(0.2))

        self.num_classes_list = num_classes_list
        self.last_dim = up_dim
        self.heads = nn.Linear(up_dim, sum(num_classes_list))

        print('Using DenseEmbedder...')
        print(f'norm: {norm}, #heads: {len(num_classes_list)}')

    def forward(self, x):
        x = x.squeeze()
        if x.ndim == 2:
            x = x[:, :, None, None]

        for layer in self.net:
            x = layer(x)

        out = x.squeeze(-1).squeeze(-1)
        out = self.heads(out)
        
        return out

################################ ArcFace Dataset ###############################

class ArcFaceDataset(Dataset):
    """
        ArcFace dataset object 
    """
    data_paths = {
        "train": os.path.join(
            os.environ["PREFGEN_ROOT"], 
            "prefgen/data/ArcFaceWSpace", 
            "train.pkl"
        ), 
        "test": os.path.join(
            os.environ["PREFGEN_ROOT"], 
            "prefgen/data/ArcFaceWSpace",
            "test.pkl"
        ),
    }

    def __init__(self, split="train", transform=None):
        """
        Args:
            split (string, optional):  Optional string saying which data split
                the dataset should come from. 
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert split in ["train", "test"]
        self.split = split
        # Check if path exists
        if os.path.exists(ArcFaceDataset.data_paths[split]):
            with open(ArcFaceDataset.data_paths[split], "rb") as f:
                self.data = pickle.load(f)
        else:
            make_arcface_dataset(
                save_path=ArcFaceDataset.data_paths[split]
            )
            with open(ArcFaceDataset.data_paths[split], "rb") as f:
                self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
            Return data 
        """
        return self.data[idx]

def load_arcface_dataloaders(batch_size=32, num_workers=4):
    """
        Loads dataloaders for arcface.  
    """
    # Load the datasets
    train_dataset = ArcFaceDataset(split="train")
    test_dataset = ArcFaceDataset(split="test")
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

def make_arcface_dataset(stylegan_generator: StyleGAN2Wrapper=None, arcface_model=None,
                            save_path="", num_examples=50000, device="cuda"):
    """
        Loads an ArcFace model and constructs a dataset mapping 
        the stylegan w-space vectors to ArcFace vectors
    """
    if stylegan_generator is None:
        # Load stylegan generator
        PRETRAINED_PATH = os.path.join(os.environ["PREFGEN_ROOT"], 'prefgen/pretrained/')
        stylegan_generator = StyleGAN2Wrapper(
            network_pkl_path=os.path.join(
                PRETRAINED_PATH, 
                'stylegan2_pt/stylegan2-ffhq-config-f.pt'
            )
        )
    if arcface_model is None:
        # Load arcface model
        arcface_model = IDLoss().to(device)
    # Data values
    data_pairs = []
    print("Making Arcface W-space dataset")
    # Iterate for num_examples
    with torch.no_grad():
        for example_index in tqdm(range(num_examples)):
            # Sample a w-vector and image from the generator
            latent_vector = stylegan_generator.randomly_sample_latent()
            w_vector = stylegan_generator.map_z_to_w(latent_vector)
            _, image = stylegan_generator.generate_image(latent_vector)
            # Embed the image using an ArcFace model
            arcface_vector = arcface_model.extract_feats(image)
            # Add the pair to the dataset
            # Add the pairs to the data_pairs
            data_pairs.append((w_vector, arcface_vector))
    # Save the dataset
    with open(save_path, "wb") as f:
        pickle.dump(data_pairs, f)

############################ Training loop for w-space to ArcFace model ###################

def train(model, args):
    """
        Trains the mapping function
    """
    print("Starting training")
    # Setup the Optimizer
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    # Setup the loss metrics
    mse_loss = torch.nn.MSELoss()
    # Load Dataloaders
    train_dataloader, test_dataloader = load_arcface_dataloaders(
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )

    best_loss = float("inf")
    # Go through the epochs
    for epoch in tqdm(range(args.epochs)):
        # Set model to train
        model.train()
        # Training Loop
        for val in tqdm(train_dataloader):
            input_w, arcface_vector = val
            model.zero_grad()
            input_w = input_w.cuda()
            arcface_vector = arcface_vector.cuda()
            # Forward pass
            predicted_arcface = model(input_w)
            # Compute loss metrics
            loss = mse_loss(predicted_arcface, arcface_vector)
            # Optimizer step
            loss.backward()
            optimizer.step()
            # Log metrics
            wandb.log({
                "Train MSE Loss": loss,
                "epoch": epoch
            })
        # Testing Loop
        # Set model to evaluation
        model.eval()
        losses = []
        with torch.no_grad():
            for input_w, arcface_vector in tqdm(test_dataloader):
                input_w = input_w.cuda()
                arcface_vector = arcface_vector.cuda()
                # Forward pass
                predicted_arcface = model(input_w)
                # Measure the triplet loss
                loss = mse_loss(predicted_arcface, arcface_vector).item()
                losses.append(loss)
                # Log metrics
                wandb.log({
                    "Test MSE Loss": loss,
                    "epoch": epoch
                })
        # Save best model
        loss = np.mean(losses)
        if loss < best_loss:
            torch.save(model.state_dict(), args.model_save_path)
            best_loss = loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training Classifiers is All You Need")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--learning_rate", default=1e-5)
    parser.add_argument("--batch_size", default=16)
    parser.add_argument("--epochs", default=20)
    parser.add_argument("--num_workers", default=0)
    PRETRAINED_PATH = os.path.join(os.environ["PREFGEN_ROOT"], 'prefgen/pretrained/')
    parser.add_argument(
        "--model_save_path", 
        default=os.path.join(
            PRETRAINED_PATH, 
            "ArcFaceWSpace",
            'identity_wspace.pt'
        )
    )

    args = parser.parse_args()

    # Setup wandb
    run = wandb.init(
        project="GanSearch",
        group="ArcFace Training",
        config=args
    )
    # Load the model
    w_space_to_arcface = DenseEmbedder(
        input_dim=512, 
        up_dim=512, 
        norm=None, 
        num_classes_list=[512]
    )
    w_space_to_arcface.to(args.device)
    w_space_to_arcface.eval()

    train(w_space_to_arcface, args)
