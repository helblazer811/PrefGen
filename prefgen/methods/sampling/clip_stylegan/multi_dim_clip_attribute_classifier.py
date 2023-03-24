import clip
import torch

from prefgen.methods.sampling.clip_stylegan.clip_attribute_classifier import CLIPAttributeClassifier, CLIPCosineSimilarityClassifier
from prefgen.methods.sampling.clip_stylegan.prompt_engineering import get_text_direction_with_prompt_engineering

class MultiDimCLIPCosineClassifier():
    """
        Classifies the attributes of a given image using the CLIP model.
    """

    def __init__(
        self, 
        generator, 
        target_prompts=None, 
        neutral_prompts=None,
        prompt_engineering=False
    ):
        self.generator = generator
        if neutral_prompts is None:
            self.neutral_prompts = [
                "a photo of a person"
                for _ in range(len(target_prompts))
            ]
        else:
            self.neutral_prompts = neutral_prompts
        self.target_prompts = target_prompts # List of lists of prompts
        self.prompt_engineering = prompt_engineering
        self.attribute_dim = len(target_prompts)
        # Make clip model to be passed to each classifier
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        # Make the attribute vector matrix
        attribute_vectors = []
        for index in range(len(self.target_prompts)):
            attribute_vector = self.compute_direction_vector(
                self.neutral_prompts[index],
                self.target_prompts[index],
                prompt_engineering=self.prompt_engineering
            ).squeeze()
            attribute_vectors.append(attribute_vector)
        self.attribute_vectors = torch.stack(attribute_vectors, dim=0)

    def compute_direction_vector(self, neutral_prompt, target_prompt, prompt_engineering=False):
        """
            Computes the direction vector for the attribute
        """
        if not prompt_engineering:
            # 1. Take text prompts and embed them in CLIP space
            tokenized_target_prompt = clip.tokenize([target_prompt]).cuda()
            target_clip_vector = self.clip_model.encode_text(tokenized_target_prompt)
            target_clip_vector = target_clip_vector / target_clip_vector.norm(dim=1, keepdim=True)
            tokenized_neutral_prompt = clip.tokenize([neutral_prompt]).cuda()
            neutral_clip_vector = self.clip_model.encode_text(tokenized_neutral_prompt)
            neutral_clip_vector = neutral_clip_vector / neutral_clip_vector.norm(dim=1, keepdim=True)
            # 2. Finds a unit vector corresponding to the difference between the 
            #  two prompt vectors 
            attribute_vector = target_clip_vector - neutral_clip_vector
            attribute_vector = attribute_vector / attribute_vector.norm(dim=1, keepdim=True)
        else:
            neutral_clip_vector = get_text_direction_with_prompt_engineering(
                neutral_prompt,
                clip_model=self.clip_model,
            )
            target_clip_vector = get_text_direction_with_prompt_engineering(
                target_prompt,
                clip_model=self.clip_model,
            )
            attribute_vector = target_clip_vector - neutral_clip_vector
            attribute_vector = attribute_vector / attribute_vector.norm(dim=1, keepdim=True)

        return attribute_vector

    def __call__(self, latent=None, image=None, preprocess=False):
        """Classify each sub attribute"""
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
        # Apply the attribute vector matrix
        attributes = torch.matmul(image_clip_vector, self.attribute_vectors.T)

        return attributes

class MultiDimCLIPAttributeClassifier():
    """
        Classifies the attributes of a given image using the CLIP model.
    """

    def __init__(self, generator, ranking_prompts=None, prompt_engineering=False):
        self.generator = generator
        self.ranking_prompts = ranking_prompts # List of lists of prompts
        self.prompt_engineering = prompt_engineering
        # Make clip model to be passed to each classifier
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        # Make each sub attribute classifier
        self.sub_classifiers = [
            CLIPAttributeClassifier(
                generator,
                self.ranking_prompts[index],
                clip_model=self.clip_model,
                preprocess=self.preprocess,
                prompt_engineering=prompt_engineering,
            )
            for index in range(len(ranking_prompts))
        ]

    def __call__(self, latent=None, image=None, preprocess=False):
        """Classify each sub attribute"""
        attributes = [
            sub_classifier(
                latent=latent, 
                image=image, 
                preprocess=preprocess
            ).unsqueeze(-1)
            for sub_classifier in self.sub_classifiers
        ]

        attributes = torch.cat(attributes, dim=-1)

        return attributes
    