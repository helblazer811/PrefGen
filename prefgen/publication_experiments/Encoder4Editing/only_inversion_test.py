from prefgen.experiments.stylegan_encoder.main import run_alignment
from prefgen.external_modules.encoder4editing.utils.common import tensor2im
from prefgen.external_modules.encoder4editing.utils.model_utils import setup_model
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import os

resize_dims = (224, 224)
image_path = "images/leo.jpg"
original_image = Image.open(image_path)
original_image = original_image.convert("RGB")
input_image = run_alignment(image_path)

def run_on_batch(inputs, net):
    images, latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
    return images, latents

def display_alongside_source_image(result_image, source_image):
    res = np.concatenate([np.array(source_image.resize(resize_dims)),
                          np.array(result_image.resize(resize_dims))], axis=1)
    return Image.fromarray(res)

input_image.resize(resize_dims)
img_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

print(os.environ["PREFGEN_ROOT"])
checkpoint = os.path.join(
    os.environ["PREFGEN_ROOT"],
    "prefgen/pretrained/stylegan2_e4e_inversion/e4e_ffhq_encode.pt"
)
print(checkpoint)
net, opts = setup_model(checkpoint, "cuda")
transformed_image = img_transforms(input_image)
with torch.no_grad():
    tic = time.time()
    images, latents = run_on_batch(transformed_image.unsqueeze(0), net)
    result_image, latent = images[0], latents[0]
    toc = time.time()
    print('Inference took {:.4f} seconds.'.format(toc - tic))
# Display inversion:
fig = plt.figure()
img = display_alongside_source_image(tensor2im(result_image), input_image)
plt.imshow(img)
plt.savefig("save_inversion_attempt.jpg")