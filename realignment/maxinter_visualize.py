import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

PLOTS_PATH = "experiments/plots/CUB/"


# =========================
# Main Function
# =========================
def main():
    # Read the CSV file
    df = pd.read_csv("results/CUB/test_maxinter.csv")

    # Set a Seaborn theme
    sns.set_theme(context="paper", style="whitegrid", palette="colorblind")

    # We call 'Baseline' UCP in the paper
    df.loc[df["model_type"] == "Baseline", "model_type"] = "UCP"


    # Create a line plot
    sns.lineplot(
        data=df, x="max_interventions", y="test_acc", hue="model_type", marker="o",linewidth=4
    )

    # Add end-to-end accuracy as "Baseline" in the plot
    plt.axhline(y=26.37,color="black", linestyle="--", linewidth=2, label="Baseline")

    # Label axes and add a title
    plt.xlabel("Maximum Interventions")
    plt.ylabel("Test Accuracy [%]")
    #plt.title("Test Accuracy vs. Max Interventions")

    # Show legend
    plt.legend(title="Model Type")

    # Display the plot
    plt.tight_layout()
    plt.savefig(PLOTS_PATH + "maxinterventions.pdf")


if __name__ == "__main__":
    main()
