"""
Exploratory Data Analysis for the telecom churn dataset.
Generates and saves plots to reports/figures/.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

DATA_PATH = "data/telecom_churn.csv"
FIG_DIR = "reports/figures"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def plot_churn_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df["Churn"].value_counts()
    colors = ["#4C72B0", "#DD8452"]
    bars = ax.bar(counts.index, counts.values, color=colors)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height}\n({height / len(df):.1%})",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
        )
    ax.set_title("Customer Churn Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of Customers")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/churn_distribution.png", dpi=150)
    plt.close(fig)


def plot_churn_by_contract(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    rate = (
        df.groupby("Contract")["Churn"]
        .apply(lambda x: (x == "Yes").mean())
        .sort_values(ascending=False)
    )
    bars = ax.bar(rate.index, rate.values, color="#C44E52")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
        )
    ax.set_title("Churn Rate by Contract Type", fontsize=14, fontweight="bold")
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Churn Rate")
    ax.set_ylim(0, max(rate.values) * 1.25)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/churn_by_contract.png", dpi=150)
    plt.close(fig)


def plot_tenure_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(
        data=df, x="tenure", hue="Churn", multiple="stack", bins=30,
        palette={"Yes": "#DD8452", "No": "#4C72B0"}, ax=ax,
    )
    ax.set_title("Tenure Distribution by Churn Status", fontsize=14, fontweight="bold")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Number of Customers")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/tenure_by_churn.png", dpi=150)
    plt.close(fig)


def plot_monthly_charges(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=df, x="Churn", y="MonthlyCharges",
        palette={"Yes": "#DD8452", "No": "#4C72B0"}, ax=ax,
    )
    ax.set_title("Monthly Charges by Churn Status", fontsize=14, fontweight="bold")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Monthly Charges ($)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/monthly_charges_by_churn.png", dpi=150)
    plt.close(fig)


def print_insights(df):
    overall_rate = (df["Churn"] == "Yes").mean()
    contract_rate = df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean())
    tenure_churn = df[df["Churn"] == "Yes"]["tenure"].median()
    tenure_stay = df[df["Churn"] == "No"]["tenure"].median()
    charges_churn = df[df["Churn"] == "Yes"]["MonthlyCharges"].median()
    charges_stay = df[df["Churn"] == "No"]["MonthlyCharges"].median()

    print("=== EDA Key Insights ===")
    print(f"Overall churn rate: {overall_rate:.2%}")
    print("\nChurn rate by contract type:")
    print(contract_rate.sort_values(ascending=False))
    print(f"\nMedian tenure - churned: {tenure_churn} months, retained: {tenure_stay} months")
    print(f"Median monthly charges - churned: ${charges_churn}, retained: ${charges_stay}")


if __name__ == "__main__":
    df = load_data()
    plot_churn_distribution(df)
    plot_churn_by_contract(df)
    plot_tenure_distribution(df)
    plot_monthly_charges(df)
    print_insights(df)
    print(f"\nSaved 4 plots to {FIG_DIR}/")
