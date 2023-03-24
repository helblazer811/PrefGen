import clip
import torch

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

def get_text_direction_with_prompt_engineering(
    prompt_text,
    clip_model=None,
):
    # Load the clip model
    if clip_model is None:
        clip_model, _ = clip.load("ViT-B/32", device="cuda")
    # Computes the mean of the prompt engineering vectors
    # for the given prompt text.
    prompt_engineering_vectors = []
    for prompt_template in prompt_templates:
        prompt = prompt_template.format(prompt_text)
        tokenized_prompt = clip.tokenize([prompt]).cuda()
        prompt_clip_vector = clip_model.encode_text(tokenized_prompt)
        prompt_clip_vector = prompt_clip_vector / prompt_clip_vector.norm(dim=1, keepdim=True)
        prompt_engineering_vectors.append(prompt_clip_vector)
    
    prompt_engineering_vector = torch.mean(
        torch.stack(prompt_engineering_vectors),
        dim=0
    )

    prompt_engineering_vector = prompt_engineering_vector / prompt_engineering_vector.norm(dim=1, keepdim=True)

    return prompt_engineering_vector