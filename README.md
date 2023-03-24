This is the code for the paper "PrefGen: Preference Guided Image Generation with Relative Attributes". 

# Setup
## Create and Activate the Conda Environment
```
    conda env create -f environment.yml
    conda activate prefgen
``` 
## Set environment variable

We need to set the root path of this repository in an environment variable.
```
    export PREFGEN_ROOT=<path to this repository>
```
## Install the package as a local pip package inside of conda
Inside of the PrefGen directory (the directory with setup.py), run the following command:
```
    pip install -e .
```
## Download models

1. Download GAN Control Model files from [here](https://drive.google.com/file/d/19v0lX69fV6zQv2HbbYUVr9gZ8ZKvUzHq/view?usp=sharing) unzip them, and put them in the folder `pretrained/stylegan2_gan_control`. Change the name of the unzipped directory to `controller_dir`.

2. Follow the directions from [here](https://github.com/rosinality/stylegan2-pytorch) to download StyleGAN2 and convert the weights to a pytorch version. Copy the weights into the `pretrained/stylegan2_pt` directory. 

3. Follow the directions from [here](https://github.com/nvlong21/Face_Recognize) to get the Face Identification model. It should have the path `model_ir_se50.pth` and put it in the directory `pretrained/face_identification`.

4. Download the model from [here](https://drive.google.com/file/d/1cUv_reLE6k3604or78EranS7XzuVMWeO/view?usp=sharing) and put it in the directory `pretrained/stylegan2_e4e_inversion`. 