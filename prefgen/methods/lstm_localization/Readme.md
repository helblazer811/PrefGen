In this directory I want to reimplement the work form "Constrained Generative Adversarial Networks for Interactive Image Generation" (https://arxiv.org/pdf/1904.02526.pdf) and set it up with our experimental infrastructure. 

# The Gist

This paper controls a GAN using feedback from pairwise comparisons by using an LSTM to filter out an estimate of user preferences, and generate an image.

- They take a GAN framework with a generator $g_\theta$ and discriminator $d_W$ and train a GAN using somethiing similar to a Wasserstein GAN optimization procedure. 
- They add an additional "constriant critic" loss that optimizes a triplet loss using the loss form the T-STE paper (t-Stochastic Triplet Embedding). 
    - They use this loss to optimize a mapping from image space to GAN latent space. 
- They then train a Constraint Generator module
    - This module has three parts: (1) a read network to learn a representaiton of each constraint (2) a process network that combines all constraints in a set into a single representation (3) a write network that maps the cosntraint to an image. 
- The process network uses an LSTM module to combine multiple constraints sequentiall into a single output representation which is then mapped to "semantic space" (aka attribute space)

# What we will do differently.

- We will not be modifying the weights of the GAN generator or discriminator as it is very expensive to train a StyleGAN2 in that way

# Plan

- Collect a dataset of paired comparisons for target attributes for FFHQ
    - Initially this can be constructed on the fly with GAN images embedded into classifier space
- Set up code to load up the attribute classifier
- Set up the architecture of the LSTM model
- Set up a train loop with the t-STE loss on the attribute space of the LSTM
- Set up the ability to measure the target/estimate erorr and plot it
- Set up a plot for evaluating the qualitative performance of final attributes
- Connect the final attribute generation to the generative model. 