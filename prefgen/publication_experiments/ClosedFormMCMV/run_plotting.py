import argparse
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'


def plot_noise_robustness(
    continuous_mcmv_data,
    mcmv_data,
    save_path="plots/noise_robustness.pdf"
):
    fig = plt.figure(figsize=(6.875/3 * 3.0, 2.6))
    f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [8, 1]}, figsize=(6.875/3 * 1.8, 2.6))

    # Make pandas dataframe out of data
    df = pd.DataFrame()
    index = 0
    for num_samples in mcmv_data.keys():
        for error in mcmv_data[num_samples]:
            df = df.append(
                pd.DataFrame(
                    {
                        "Number of Query Samples": num_samples,
                        "Preference Estimate\nMSE to Target": error,
                        "Method": "Active Finite"
                    },
                    index=[index]
                )
            )
            index += 1
    # Plot violin plots
    # set color palette
    # sns.set_palette("classic")
    left_plot = sns.boxplot(
        df,
        x="Number of Query Samples",
        y="Preference Estimate\nMSE to Target",
        hue="Method",
        palette={
        #     "LSTM Baseline": plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
            "Active Finite": plt.rcParams['axes.prop_cycle'].by_key()['color'][1],
        },
        ax=a0,
    )
    left_plot.legend_.remove()
    a0.legend(
        loc='lower right',
        handles=[
            mpl.patches.Patch(
                color=sns.color_palette("deep")[1],
                label="Active Querying (Best of N)"
            ),
            mpl.patches.Patch(
                color=sns.color_palette("deep")[0],
                label="Active Querying (Closed Form)"
            ),
        ]
    )
    a1.axes.get_xaxis().set_visible(False)

    df = pd.DataFrame()
    index = 0
    for error in continuous_mcmv_data:
        df = df.append(
            pd.DataFrame(
                {
                    "Preference Estimate MSE to Target": error,
                    "Method": "Active Continuous"
                },
                index=[index]
            )
        )
        
    plt.yscale("log")
    sns.boxplot(
        df,
        ax=a1,
    )
    sns.set(font_scale=1.2)
    a0.set_yscale("log")
    a1.set_yscale("log")
    a1.set_yticks([])
    a1.set_yticklabels([])
    plt.tight_layout()
    a1.set_ylim(bottom=0.00003, top=0.01)
    a0.set_ylim(bottom=0.00003, top=0.01)
    
    # plt.ylim(bottom=0.000001, top=0.0)
    plt.savefig(save_path, dpi=200)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mcmv_data_path", 
        type=str, 
        default="data/mcmv_different_num_samples_pretty_good.pkl"
    )
    parser.add_argument(
        "--continuous_mcmv_data_path", 
        type=str, 
        default="data/continuous_mcmcv_data.pkl"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="plots/final_performance_vs_num_samples.pdf"
    )

    args = parser.parse_args()

    mcmv_data = pickle.load(open(args.mcmv_data_path, "rb"))
    continuous_mcmv_data = pickle.load(open(args.continuous_mcmv_data_path, "rb"))

    plot_noise_robustness(
        continuous_mcmv_data,
        mcmv_data,
        save_path=args.save_path
    )