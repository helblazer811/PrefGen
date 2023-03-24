"""
    This is where I implement the conditional energy function necessary to 
    produce a function proportional to p(z|a) so that we can guide the LD
    process in the direction of the conditional score log grad p(z|a). 
"""

import torch

def compute_continuous_conditional_energy(current_latent, current_attributes, target_attributes,
                                            energy_sigma=0.1):
    """
        This function computes the continuous conditional energy 
        from the LACE paper. 
    """
    assert current_attributes.shape == target_attributes.shape, \
        f"current attributes shape: {current_attributes.shape}, target_attributes shape: {target_attributes.shape}"
    cond_energy = torch.linalg.norm(
        current_attributes - target_attributes
    ) ** 2 * 0.5 / (energy_sigma ** 2)
    # Subtract L2 norm
    l2_norm = torch.linalg.norm(current_latent) ** 2 * 0.5
    # l2_norm = 0.0

    total_energy = cond_energy + l2_norm
    # print(f"Cond energy: {cond_energy}, L2 norm : {l2_norm}")

    # assert torch.isfinite(single_cond_energy) and not torch.isnan(single_cond_energy)
    return total_energy

def compute_discrete_conditional_energy(current_latent, current_attributes, 
                                        target_attributes, dis_temp=1.0):
    """
        Computes conditional energy for discrete attributes 
    """
    single_cond_energy = torch.gather(current_attributes / dis_temp, 0, target_attributes).squeeze() - \
                        current_attributes.logsumexp(1)
    # Subtract L2 norm
    single_cond_energy = torch.linalg.norm(current_latent) ** 2 * 0.5

    return single_cond_energy