import os
import argparse
import pickle
import torch
from tqdm import tqdm

from prefgen.methods.datasets.lstm import ConstraintDataset
from prefgen.methods.lstm_localization.model import PreferenceLSTM

def evaluate_lstm_model_performance(
    model: PreferenceLSTM, 
    test_dataset, 
    num_examples, 
    num_queries, 
    query_testing_interval,
):
    save_data = {}
    # For each number of query intervals
    for num_queries in range(query_testing_interval, num_queries + 1, query_testing_interval):
        num_queries_data = []
        # For each example in the first `num_examples` examples
        for example_idx in tqdm(range(num_examples)):
            # Select the queries [0: num_queries]
            constraint_attributes = test_dataset[example_idx][-1]
            constraint_attributes = constraint_attributes[0:num_queries, :]
            constraint_attributes = torch.Tensor(constraint_attributes).unsqueeze(0)
            # Predict using LSTM the posterior estimate
            preference_estimate = model.compute_lstm_rollout(
                constraint_attributes,
            )
            preference_estimate = preference_estimate.detach().cpu().numpy()
            # Save the example index, the number of queries used, and the final estimate
            num_queries_data.append({
                "example_idx": example_idx,
                "preference_estimate": preference_estimate
            })
        save_data[num_queries] = num_queries_data

    return save_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attribute_names",
        default=["yaw", "age"]
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
        default="data/lstm_data_20q.pkl"
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
    model.load_state_dict(torch.load(args.model_save_path))
    save_data = evaluate_lstm_model_performance(
        model, 
        test_dataset, 
        args.num_examples, 
        args.num_queries, 
        args.query_testing_interval,
    )

    with open(args.lstm_data_path, "wb") as f:
        pickle.dump(save_data, f)