"""
    In this directory I want to implement a CLIP model that
    takes in a bunch of target prompts and a bunch of neutral
    prompts for a given set of attributes, embeds them into clip
    space using prompt engineering to produce a bunch of vectors,
    and then uses PCA to reduce the dimensionality of the vectors
    to finally produce an embedding where triplet loss and other metrics
    can be evaluated. 

    This model is basicaly a multi dimensional clip cosine similarity
    model, where the directions are computed using PCA. 
"""
import clip
from sklearn.decomposition import PCA
import torch
import numpy as np

prompt_templates = [
    'a bad photo of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'a low resolution photo of a {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a photo of a nice {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a good photo of a {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a dark photo of a {}.',
    'graffiti of the {}.',
]

class CLIPPCAAttributeClassifier():

    def __init__(
        self, 
        generator,
        neutral_prompts,
        target_prompts,
        clip_model=None, 
        preprocess=None,
        pca_components=10,
    ):
        self.generator = generator
        self.neutral_prompts = neutral_prompts
        self.target_prompts = target_prompts
        self.pca_components = pca_components
        # Load the clip model
        if clip_model is None or preprocess is None:
            self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        else:
            self.clip_model = clip_model
            self.preprocess = preprocess

        # Construct basis vectors
        self.basis_vectors = self._construct_basis_vectors(
            pca_components=self.pca_components
        )

    def _construct_basis_vectors(self, pca_components=10):
        difference_vectors = []
        # Go through each attribute
        for attribute_index in range(len(self.neutral_prompts)):
            # For each prompt compute the neutral and target vectors
            for prompt_template in prompt_templates:
                # Compute the neutral vector
                neutral_text = prompt_template.format(self.neutral_prompts[attribute_index])
                neutral_text = clip.tokenize([neutral_text]).cuda()
                with torch.no_grad():
                    neutral_vector = self.clip_model.encode_text(neutral_text)
                # Compute the target vector
                target_text = prompt_template.format(self.target_prompts[attribute_index])
                target_text = clip.tokenize([target_text]).cuda()
                with torch.no_grad():
                    target_vector = self.clip_model.encode_text(target_text)
                # Compute the difference vector
                difference_vector = target_vector - neutral_vector
                difference_vector = difference_vector / torch.norm(difference_vector)
                difference_vector = difference_vector.detach().cpu().numpy()
                difference_vectors.append(difference_vector)
        # Compute the PCA with sklearn
        difference_vectors = np.stack(difference_vectors).squeeze()
        pca = PCA(n_components=pca_components)
        pca.fit(difference_vectors)
        basis_vectors = pca.components_
        basis_vectors = torch.from_numpy(basis_vectors).half().cuda()

        return basis_vectors

    def __call__(self, latent=None, image=None, preprocess=False):
        if not latent is None:
            _, image = self.generator.generate_image(latent=latent)
            # Convert the torch image to a pil image
        if preprocess:
            image = self.preprocess(image).cuda()
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
        # NOTE: It is important to ensure we preprocess the image in a way
        # that does not break the compuation graph, thus allowing gradients
        # to flow properly. 
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(
            kernel_size=image.shape[-1] // 32
        )
        image = self.avg_pool(self.upsample(image))
        # Compute the clip vector for the image
        image_clip_vector = self.clip_model.encode_image(image)
        image_clip_vector = image_clip_vector / torch.norm(image_clip_vector)
        # Compute the dot product between the attribute vector and 
        # each PCA basis vector
        attributes = torch.matmul(image_clip_vector, self.basis_vectors.T)

        return attributes