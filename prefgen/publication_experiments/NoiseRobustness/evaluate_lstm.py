import argparse
import numpy as np
import os
import argparse
import pickle
import torch
from tqdm import tqdm
import random

from prefgen.methods.datasets.lstm import ConstraintDataset
from prefgen.methods.lstm_localization.model import PreferenceLSTM

def randomly_flip_constraints(constraints, flip_chance):
    constraint_dim = constraints.shape[1]
    attribute_dim = int(constraint_dim / 2)
    for constraint_idx in range(constraints.shape[0]):
        flip = random.random() < flip_chance
        if flip:
            positive = constraints[constraint_idx, 0:attribute_dim]
            negative = constraints[constraint_idx, attribute_dim:]
            constraints[constraint_idx, 0:attribute_dim] = negative
            constraints[constraint_idx, attribute_dim:] = positive

def evaluate_lstm_model_performance(
    model: PreferenceLSTM, 
    test_dataset, 
    num_examples, 
    flip_chances=None
):
    save_data = {}
    for flip_chance in flip_chances:
        final_errors = []
        # For each example in the first `num_examples` examples
        for example_idx in tqdm(range(num_examples)):
            # Select the queries [0: num_queries]
            constraint_attributes = test_dataset[example_idx][-1]
            # Randomly flip
            randomly_flip_constraints(
                constraint_attributes, 
                flip_chance=flip_chance
            )
            constraint_attributes = torch.Tensor(constraint_attributes).unsqueeze(0)
            # Predict using LSTM the posterior estimate
            preference_estimate = model.compute_lstm_rollout(
                constraint_attributes,
            )
            preference_estimate = preference_estimate.detach().cpu().numpy()
            final_error = np.linalg.norm(
                preference_estimate - test_dataset[example_idx][1]
            ) ** 2
            # Save the example index, the number of queries used, and the final estimate
            final_errors.append(
                final_error
            )

        save_data[flip_chance] = final_errors

    return save_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--attribute_names", default=["yaw", "age"])
    parser.add_argument(
        "--flip_chances", 
        default=[
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5
        ]
    )
    parser.add_argument(
        "--test_save_path", 
        default=os.path.join(
            os.environ["PREFGEN_ROOT"],
            "prefgen/data/lstm_dataset/test_dataset_gan_control.pkl"
        ),
    )
    parser.add_argument(
        "--lstm_data_path", 
        type=str, 
        default="data/lstm_data_random_flip_2d.pkl"
    )
    parser.add_argument(
        "--num_examples",
        default=20
    )
    parser.add_argument(
        "--num_queries",
        default=30
    )
    parser.add_argument(
        "--query_testing_interval",
        default=1
    )
    parser.add_argument(
        "--model_save_path", 
        default=os.path.join(
            os.environ["PREFGEN_ROOT"],
            "prefgen/pretrained/lstm_models/lstm_model.pth"
        ),
    )

    args = parser.parse_args()
    # Make the dataset
    test_dataset = ConstraintDataset(args.test_save_path)
    # Load model
    model = PreferenceLSTM(
        attribute_size=2,
    )
    model.load_state_dict(
        torch.load(args.model_save_path)
    )
    save_data = evaluate_lstm_model_performance(
        model, 
        test_dataset, 
        args.num_examples, 
        flip_chances=args.flip_chances
    )

    with open(args.lstm_data_path, "wb") as f:
        pickle.dump(save_data, f)