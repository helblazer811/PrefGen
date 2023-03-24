import argparse
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'


def plot_noise_robustness(
    lstm_data,
    mcmv_data,
    save_path="plots/noise_robustness.pdf"
):
    fig = plt.figure(figsize=(6.875/3 * 1.8, 3.0))
    # Make pandas dataframe out of data
    df = pd.DataFrame()
    index = 0
    for noise_level in lstm_data.keys():
        for error in lstm_data[noise_level]:
            df = df.append(
                pd.DataFrame(
                    {
                        "Random Query Flip Probability": noise_level,
                        "Preference Estimate MSE to Target": error,
                        "Method": "LSTM Baseline"
                    },
                    index=[index]
                )
            )
            index += 1
    for noise_level in mcmv_data.keys():
        for error in mcmv_data[noise_level]:
            df = df.append(
                pd.DataFrame(
                    {
                        "Random Query Flip Probability": noise_level,
                        "Preference Estimate MSE to Target": error,
                        "Method": "Ours Active"
                    },
                    index=[index]
                )
            )
            index += 1
    # Plot violin plots
    # set color palette
    # sns.set_palette("classic")
    sns.boxplot(
        df,
        x="Random Query Flip Probability",
        y="Preference Estimate MSE to Target",
        hue="Method",
        palette={
            "LSTM Baseline": plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
            "Ours Active": plt.rcParams['axes.prop_cycle'].by_key()['color'][1],
        }
    )

    sns.set(font_scale = 1.2)
    plt.yscale("log")
    plt.tight_layout()
    plt.ylim(bottom=0.000001, top=0.0)
    plt.savefig(save_path, dpi=200)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lstm_data_path", 
        type=str, 
        default="data/lstm_data_random_flip_2d.pkl"
    )
    parser.add_argument(
        "--mcmv_data_path", 
        type=str, 
        default="data/mcmv_data_random_flip_2d.pkl"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="plots/noise_robustness.pdf"
    )

    args = parser.parse_args()

    lstm_data = pickle.load(open(args.lstm_data_path, "rb"))
    mcmv_data = pickle.load(open(args.mcmv_data_path, "rb"))

    plot_noise_robustness(
        lstm_data,
        mcmv_data,
        save_path=args.save_path
    )