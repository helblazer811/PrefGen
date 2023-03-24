import clip
import torch
import torchvision.transforms as T
from sklearn.decomposition import PCA
from tqdm import tqdm

from prefgen.methods.sampling.clip_stylegan.prompt_engineering import get_text_direction_with_prompt_engineering

class CLIPCosineSimilarityClassifier():
    """
        This class handles constructing zero-shot
        attribute classifiers given just text prompts.  
    """

    def __init__(
        self, 
        generator,
        neutral_prompt="a person",
        target_prompt="an angry person",
        clip_model=None,
        preprocess=None,
        prompt_engineering=False,
        scale_output=False,
        value_range=(-1, 1),
    ):
        self.generator = generator
        self.neutral_prompt = neutral_prompt
        self.target_prompt = target_prompt
        self.prompt_engineering = prompt_engineering
        self.scale_output = scale_output
        self.value_range = value_range
        # Load the clip model
        if clip_model is None or preprocess is None:
            self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        else:
            self.clip_model = clip_model
            self.preprocess = preprocess
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.attribute_vector = self.compute_direction_vector(
            prompt_engineering=prompt_engineering
        )
        self.attribute_vector = torch.nn.Parameter(
            self.attribute_vector,
            requires_grad=True
        ).cuda()
    
    def compute_direction_vector(self, prompt_engineering=False):
        """
            Computes the direction vector for the attribute
        """
        if not prompt_engineering:
            # 1. Take text prompts and embed them in CLIP space
            tokenized_target_prompt = clip.tokenize([self.target_prompt]).cuda()
            target_clip_vector = self.clip_model.encode_text(tokenized_target_prompt)
            target_clip_vector = target_clip_vector / target_clip_vector.norm(dim=1, keepdim=True)
            tokenized_neutral_prompt = clip.tokenize([self.neutral_prompt]).cuda()
            neutral_clip_vector = self.clip_model.encode_text(tokenized_neutral_prompt)
            neutral_clip_vector = neutral_clip_vector / neutral_clip_vector.norm(dim=1, keepdim=True)
            # 2. Finds a unit vector corresponding to the difference between the 
            #  two prompt vectors 
            attribute_vector = target_clip_vector - neutral_clip_vector
            attribute_vector = attribute_vector / attribute_vector.norm(dim=1, keepdim=True)
        else:
            neutral_clip_vector = get_text_direction_with_prompt_engineering(
                self.neutral_prompt,
                clip_model=self.clip_model,
            )
            target_clip_vector = get_text_direction_with_prompt_engineering(
                self.target_prompt,
                clip_model=self.clip_model,
            )
            attribute_vector = target_clip_vector - neutral_clip_vector
            attribute_vector = attribute_vector / attribute_vector.norm(dim=1, keepdim=True)

        return attribute_vector

    def __call__(self, latent=None, image=None, preprocess=False):
        """Performs the attribute classification"""
        if not latent is None:
            _, image = self.generator.generate_image(latent=latent)
            # Convert the torch image to a pil image
        if preprocess:
            image = self.preprocess(image).cuda()
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
        # print(f"Image shape: {image.shape}")
        image_size = image.shape[-1]
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(
            kernel_size=image_size // 32
        )
        # NOTE: It is important to ensure we preprocess the image in a way
        # that does not break the compuation graph, thus allowing gradients
        # to flow properly. 
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = self.avg_pool(self.upsample(image))
        # Compute the clip vector for the image
        image_clip_vector = self.clip_model.encode_image(image)
        # Compute the dot product between the attribute vector and the image
        attribute_value = self.cos(image_clip_vector, self.attribute_vector)
        # Clip values to range, and stretch to [-1, 1]
        attribute_value = torch.clip(
            attribute_value,
            min=self.value_range[0],
            max=self.value_range[1]
        )
        # Rescale the output
        if self.scale_output:
            attribute_value = (attribute_value - self.value_range[0]) / (self.value_range[1] - self.value_range[0])
            
        return attribute_value

class CLIPAttributeClassifier():
    """
        This class handles constructing zero-shot
        attribute classifiers given just text prompts.  
    """

    def __init__(
        self, 
        generator, 
        rank_prompts=[
            "a child",
            "a teenager",
            "a young adult",
            "a middle aged person",
            "an old person",
            "a very old person"
        ],
        clip_model=None,
        preprocess=None,
        project_to_face_subspace=False,
        prompt_engineering=False
    ):
        self.generator = generator
        self.rank_prompts = rank_prompts
        self.project_to_face_subspace = project_to_face_subspace
        self.prompt_engineering = prompt_engineering
        # Make the clip model
        if clip_model is None or preprocess is None:
            self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        else:
            self.clip_model, self.preprocess = clip_model, preprocess
        # Make the face subspace projection matrix
        if self.project_to_face_subspace:
            self.face_subspace_matrix = self.make_face_subspace_projection_matrix()
        # Compute ranking vectors
        self.ranking_vectors = self.compute_ranking_vectors(
            prompt_engineering=self.prompt_engineering
        )

    def compute_ranking_vectors(self, prompt_engineering=False):
        """
            Computes the ranking vectors for each prompt.
        """
        if prompt_engineering: 
            with torch.no_grad():
                ranking_vectors = []
                for prompt in self.rank_prompts:
                    prompt_clip_vector = get_text_direction_with_prompt_engineering(
                        prompt,
                        clip_model=self.clip_model,
                    )
                    ranking_vectors.append(prompt_clip_vector)
                ranking_vectors = torch.stack(ranking_vectors).squeeze()

                return ranking_vectors
        else:
            tokenized_prompt = clip.tokenize(self.rank_prompts).cuda()
            rank_vectors = self.clip_model.encode_text(tokenized_prompt)
            rank_vectors = rank_vectors / rank_vectors.norm(dim=-1, keepdim=True)
            return rank_vectors

    def make_face_subspace_projection_matrix(
        self, 
        dimension=50,
        num_examples=500,
        projection_type="pca"
    ):
        """
            Makes a projection matrix that maps CLIP vectors to a face
            image subspace.
        """
        with torch.no_grad():
            print("Constructing face subspace projection")
            clip_vectors = []
            for _ in tqdm(range(num_examples)):
                _, image = self.generator.generate_image()
                self.upsample = torch.nn.Upsample(scale_factor=7)
                self.avg_pool = torch.nn.AvgPool2d(kernel_size=image.shape[-1] // 32)
                # Preprocess the image
                image = self.avg_pool(self.upsample(image))
                # Compute the clip vector for the image
                image_clip_vector = self.clip_model.encode_image(image)
                clip_vectors.append(image_clip_vector)
            # Make the projection matrix
            if projection_type == "pca":
                clip_vectors = torch.cat(clip_vectors, dim=0)
                clip_vectors = clip_vectors.cpu().numpy()
                pca = PCA(n_components=dimension)
                pca.fit(clip_vectors)
                projection_matrix = torch.from_numpy(pca.components_).cuda()
                projection_matrix = projection_matrix.half()
                return projection_matrix
            elif projection_type == "gram_schmidt":
                raise NotImplementedError()
            else:
                raise ValueError("Invalid projection type.")


    def __call__(self, latent=None, image=None, preprocess=False, return_probs=False):
        """
            Use CLIP to predict the age of an image.
        """
        if not latent is None:
            assert not self.generator is None, "Must provide a generator if latent is provided."
            _, image = self.generator.generate_image(latent=latent)
            self.upsample = torch.nn.Upsample(scale_factor=7)
            self.avg_pool = torch.nn.AvgPool2d(kernel_size=image.shape[-1] // 32)
            # Preprocess the image
            image = self.avg_pool(self.upsample(image))
        if preprocess:
            image = self.preprocess(image).cuda()
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
        # Don't use gradients
        with torch.no_grad():
            # Make rank vectors
            image_features = self.clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            rank_features = self.ranking_vectors
            # Apply face subspace projection
            if self.project_to_face_subspace:
                image_features = image_features @ self.face_subspace_matrix.t()
                rank_features = rank_features @ self.face_subspace_matrix.t()
            # Compute the rank logits
            logit_scale = self.clip_model.logit_scale.exp()
            dot_products = torch.matmul(
                rank_features,
                image_features.t()
            ).squeeze(-1)
            logits_per_image = logit_scale * dot_products
            """
            logits_per_image, _ = self.clip_model(image, text)
            """
            probs = logits_per_image.softmax(dim=-1)
            # Compute the age prediction
            # by taking the weighted average
            # Make the weights
            weights = torch.arange(len(self.rank_prompts)) / len(self.rank_prompts)
            weights = weights.to("cuda")
            attribute_score = torch.sum(weights * probs)
            # attribute_score = attribute_score.item()
            
            if return_probs:
                return attribute_score, probs
            else:
                return attribute_score