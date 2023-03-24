"""
    I need to be able to generate from StyleGAN2 so this creates a nice wrapper
    arround the StyleGAN2 model so that I can generate images given latent vectors
    as well as sample randomly from StyleGAN2 latent space. 
"""
import os
import torch
import numpy as np
from torch.nn import Module
from prefgen.methods.generative_models.stylegan2 import Generator

stylegan_pth = os.path.join(
    os.environ["PREFGEN_ROOT"], 
    "prefgen/pretrained/stylegan2_pt/stylegan2-ffhq-config-f.pt"
)

class StyleGAN2Wrapper(Module):
    """
        Wrapper object around stylegan
    """

    def __init__(
        self, 
        network_pkl_path=stylegan_pth, 
        latent_dim=512, 
        num_mlp=8, 
        image_size=1024,
        device="cuda", 
        w_space_latent=True, 
        extended_latent=False,
        rescale=True
    ):
        super(StyleGAN2Wrapper, self).__init__()
        self.latent_dim = latent_dim
        self.num_mlp = num_mlp
        self.image_size = image_size
        self.w_space_latent = w_space_latent
        self.extended_latent = extended_latent
        # Initialize the Generator
        generator = Generator(
            size=image_size, 
            style_dim=latent_dim, 
            n_mlp=num_mlp
        ).to(device)
        assert os.path.exists(network_pkl_path), network_pkl_path
        checkpoint = torch.load(network_pkl_path, map_location=torch.device(device))
        generator.load_state_dict(checkpoint["g_ema"])
        self.generator = generator
        self.mean_w = self.generator.mean_latent(n_latent=2048)
        self.rescale = rescale

    def randomly_sample_latent(
        self, 
        device="cuda", 
        w_space_latent=None, 
        extended_latent=None,
        seed=None
    ):
        """Randomly sample a latent vector"""
        if not seed is None:
            torch.manual_seed(seed)
            
        z_init = torch.FloatTensor(
            1, 
            self.latent_dim
        ).normal_(0, 1).to(device)
        z_init.requires_grad = True
        
        if extended_latent is None:
            extended_latent = self.extended_latent
        if w_space_latent is None:
            w_space_latent = self.w_space_latent

        if extended_latent:
            w_vectors = self.map_z_to_w(z_init)
            w_vectors = w_vectors.unsqueeze(1).repeat([1, 18, 1])  # bs x n_mappings x 512
            return w_vectors

        if w_space_latent:
            return self.map_z_to_w(z_init)
        else:
            return z_init

    def map_z_to_w(
        self, 
        z, 
        truncation=0.7
    ):
        """Map z to w"""
        assert z.shape[-1] == self.generator.style_dim, z.shape
        w = self.generator.style(z)  # no truncation

        if truncation < 1:
            truncation_latent = self.mean_w.to(z.device)
            w = truncation_latent + truncation * (w - truncation_latent)

        return w

    def generate_image(
        self, 
        latent=None, 
        w_space_latent=None, 
        extended_latent=None,
        randomize_noise=True,
        seed=None
    ):
        """Decode a latent vector into an image"""
        if w_space_latent is None:
            w_space_latent = self.w_space_latent
        if extended_latent is None:
            extended_latent = self.extended_latent
            
        if latent is None:
            latent = self.randomly_sample_latent(seed=seed) # latent z vector
        
        if isinstance(latent, np.ndarray):
            latent = torch.Tensor(latent).cuda()

        if not extended_latent: 
            if not w_space_latent:
                w_vectors = self.map_z_to_w(latent)
            else:
                w_vectors = latent

            if w_vectors.ndim == 2:
                w_vectors = w_vectors.unsqueeze(1).repeat([1, 18, 1])  # bs x n_mappings x 512
        else:
            w_vectors = latent
            
        image, _ = self.generator(
            [w_vectors],
            randomize_noise=randomize_noise, 
            input_is_latent=True
        )
        # Rescale image
        if self.rescale:
            image = (image - torch.min(image)) / torch.max(image - torch.min(image))

        return latent, image