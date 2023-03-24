"""
    Here we have a dataset composed of ground truth PIL images
    and their respective encoded latent vectors. 
"""
import torch
import os
from PIL import Image

# Make a torch dataset
class E4EDataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        image_path=os.path.join(
            os.environ["PREFGEN_ROOT"],
            "prefgen/methods/encoder/encoder4editing/images"
        ),
        latent_path=os.path.join(
            os.environ["PREFGEN_ROOT"],
            "prefgen/methods/encoder/encoder4editing/save/latents.pt"
        )
    ):
        self.image_path = image_path
        self.image_paths = os.listdir(image_path)
        self.latents = torch.load(latent_path).to("cuda")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = os.path.join(
            self.image_path,
            self.image_paths[index]
        )
        image = Image.open(image_path)
        latent = self.latents[index]

        return image, latent