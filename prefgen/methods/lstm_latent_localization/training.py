from prefgen.methods.generative_models.stylegan2.stylegan_wrapper import StyleGAN2Wrapper
from prefgen.methods.lstm_latent_localization.model import PreferenceLatentLSTM
from prefgen.methods.lstm_localization.model import PreferenceLSTM
from prefgen.methods.datasets.lstm import ConstraintDataset
from prefgen.methods.plotting.localization import compute_attribute_rank_loss
from prefgen.methods.sampling.langevin_dynamics.classifiers.ffhq_classifier.load import load_ffhq_wspace_classifier
import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm
import wandb

def stochastic_triplet_embedding_loss(anchor, positive, negative, alpha=1):
    """This computes the loss from the T-STE paper over a given triplet"""
    # Compute the distances
    anchor_positive_dist = torch.linalg.norm(anchor - positive, dim=-1)
    anchor_negative_dist = torch.linalg.norm(anchor - negative, dim=-1)
    # Compute each of the terms
    anchor_positive_term = (1 + (anchor_positive_dist ** 2) / alpha) ** (-1*(alpha + 1) / 2)
    anchor_negative_term = (1 + (anchor_negative_dist ** 2) / alpha) ** (-1*(alpha + 1) / 2)
    # Compute the loss
    return (anchor_positive_term) / (anchor_negative_term + anchor_positive_term)

def constraint_loss(preference_estimate, constraint_attributes):
    loss = 0.0
    batch_size = constraint_attributes.shape[0]
    num_constraints = constraint_attributes.shape[1]
    attribute_dim = int(constraint_attributes.shape[-1] / 2)
    # Flatten constraints
    # Evaluate constraint loss on all constraints
    constraint_attributes = torch.reshape(
        constraint_attributes,
        (batch_size * num_constraints, -1)
    ).cuda()
    # Also reshape preference estimate
    # Copy preference estimate over the num constraints as it is the same for each
    preference_estimate = preference_estimate.unsqueeze(1).repeat(
        1, num_constraints, 1
    )
    preference_estimate = torch.reshape(
        preference_estimate,
        (batch_size * num_constraints, -1)
    ).cuda()
    # Pull out the constraint positive and negatives
    constraint_positive = constraint_attributes[:, 0:attribute_dim]
    constraint_negative = constraint_attributes[:, attribute_dim:]
    assert constraint_positive.shape == constraint_negative.shape
    assert preference_estimate.shape == constraint_negative.shape, (preference_estimate.shape, constraint_negative.shape)
    
    constraint_losses = stochastic_triplet_embedding_loss(
        preference_estimate,
        constraint_positive,
        constraint_negative
    )

    constraint_loss = torch.mean(constraint_losses)
    
    return (-1 / (num_constraints * batch_size)) * constraint_loss

def run_testing(model, latent_sampler, test_dataset, batch_size=32, num_lstm_iterations=5):
    """Runs a test loop"""
    # Evaluate the percentage of constraints satisfied in the test set
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )
    is_positive = []
    percentages_closer_to_target = []

    ranking_attribute_samples = latent_sampler.randomly_sample_attributes(
        500,
        return_dict=False
    ).detach().cpu().numpy()

    # Iterate through the batches
    for target_latent, target_attributes, constraint_attributes, initial_latent in test_dataloader:
        if isinstance(model, PreferenceLSTM):
            preference_estimate = model.compute_lstm_rollout(
                constraint_attributes,
                num_iterations=num_lstm_iterations
            )
        else:
            preference_estimate, _ = model.compute_lstm_rollout(
                initial_latent,
                constraint_attributes,
                num_iterations=num_lstm_iterations
            )
        # Evaluate percentages closer to target
        for example_index in range(batch_size):
            last_estimate = preference_estimate[example_index]
            ranked_attribute_distance = compute_attribute_rank_loss(
                latent_sampler=latent_sampler,
                current_attributes=last_estimate.detach().cpu().numpy(),
                target_attributes=target_attributes[example_index].detach().cpu().numpy(),
                attribute_samples=ranking_attribute_samples
            )
            percentages_closer_to_target.append(ranked_attribute_distance)
        # Evaluate what percentage of the constraints are staisfied 
        # if the preference_estimate is subbed in for the anchor
        # 
        batch_size = constraint_attributes.shape[0]
        num_constraints = constraint_attributes.shape[1]
        attribute_dim = int(constraint_attributes.shape[-1] / 2)
        # Flatten constraints
        # Evaluate constraint loss on all constraints
        constraint_attributes = torch.reshape(
            constraint_attributes,
            (batch_size * num_constraints, -1)
        ).cuda()
        # Also reshape preference estimate
        # Copy preference estimate over the num constraints as it is the same for each
        preference_estimate = preference_estimate.unsqueeze(1).repeat(
            1, num_constraints, 1
        )
        preference_estimate = torch.reshape(
            preference_estimate,
            (batch_size * num_constraints, -1)
        ).cuda()
        # Pull out the constraint positive and negatives
        constraint_positive = constraint_attributes[:, 0:attribute_dim]
        constraint_negative = constraint_attributes[:, attribute_dim:]
        assert constraint_positive.shape == constraint_negative.shape
        assert preference_estimate.shape == constraint_negative.shape, (preference_estimate.shape, constraint_negative.shape)
        # Compute the distances
        anchor_positive_dist = torch.linalg.norm(
            preference_estimate - constraint_positive, 
            dim=-1
        )
        anchor_negative_dist = torch.linalg.norm(
            preference_estimate - constraint_negative, 
            dim=-1
        )
        constraint_bool = (anchor_positive_dist < anchor_negative_dist).tolist()
        is_positive.extend(constraint_bool)
        
    percentage_satisfied = np.sum(is_positive) / len(is_positive)
    percentage_closer_to_target = np.mean(percentages_closer_to_target)

    return percentage_satisfied, percentage_closer_to_target

def train(
    model,
    latent_sampler,
    train_dataset, 
    test_dataset, 
    num_epochs=20, 
    num_lstm_iterations=10,
    batch_size=32,
    model_save_path=""
):
    """Runs the train loop for a certain amount of iterations"""
    if isinstance(model, PreferenceLatentLSTM):
        model_is_preference_latent = True
    else:
        model_is_preference_latent = False
    # Start wandb
    run = wandb.init(
        project="GanSearch",
        group="LSTMTesting",
    )
    # NOTE: doing one epoch for testing purposes
    optimizer = torch.optim.Adam(
        model.parameters(), 
        betas=(0, 0.9),
        lr=0.0001
    )
    # SEtup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1
    )
    # Make a dataloader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size
    )
    
    best_test_percentage_closer = 100.0

    for epoch in tqdm(range(num_epochs)):
        losses = []
        for target_latent, target_attributes, constraints, initial_latent in tqdm(train_dataloader):
            optimizer.zero_grad()
            # Randomly choose a number of constraints to evaluate
            num_constraints = constraints.shape[1]
            num_to_evaluate = np.random.randint(1, num_constraints + 1)
            # Randomly choose a subset of the constraints
            random_indices = np.random.choice(
                np.arange(num_constraints),
                size=num_to_evaluate
            )
            constraints = constraints[:, random_indices, :]
            # Do model forward pass on constraints
            if model_is_preference_latent:
                preference_estimate, model_out = model.compute_lstm_rollout(
                    initial_latent, 
                    constraints,
                    num_iterations=num_lstm_iterations
                )
            else:
                preference_estimate = model.compute_lstm_rollout(
                    constraints,
                    num_iterations=num_lstm_iterations
                )
            # print(f"Predicted preference estimate: {preference_estimate}")
            # print(f"Target attributes: {target_attributes}")
            # Compute loss for preference_estimate
            loss = constraint_loss(
                preference_estimate, 
                constraints
            )

            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        # scheduler.step()

        # Run the testing
        percentage_satisfied_test, percentage_closer_test = run_testing(
            model, 
            latent_sampler,
            test_dataset, 
            batch_size=batch_size,
            num_lstm_iterations=num_lstm_iterations
        )
        percentage_satisfied_train, percentage_closer_train = run_testing(
            model, 
            latent_sampler,
            train_dataset, 
            batch_size=batch_size,
            num_lstm_iterations=num_lstm_iterations
        )
        """
        if percentage_closer_test < best_test_percentage_closer:
            best_test_percentage_closer = percentage_closer_test
            torch.save(model.state_dict(), model_save_path)
        """

        wandb.log({
            "loss": np.mean(losses),
            "percentage_satisfied_test": percentage_satisfied_test,
            "percentage_satisfied_train": percentage_satisfied_train,
            "percentage_closer_test": percentage_closer_test,
            "percentage_closer_train": percentage_closer_train,
            "epoch": epoch
        })

