import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Set modern aesthetics
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16
    })

    # Create plots directory if it doesn't exist
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    print(f"[*] Created directory '{plots_dir}' for saving figures.")

    # 1. Load the dataset
    print("[*] Loading creditcard.csv...")
    try:
        df = pd.read_csv("data/creditcard.csv")
    except FileNotFoundError:
        print("[!] Error: creditcard.csv not found in the current directory.")
        return
    
    # 2. Basic dataset statistics
    n_rows, n_cols = df.shape
    n_fraud = df[df['Class'] == 1].shape[0]
    n_normal = df[df['Class'] == 0].shape[0]
    pct_fraud = (n_fraud / n_rows) * 100

    print("\n" + "="*50)
    print("                DATASET SUMMARY")
    print("="*50)
    print(f"Total Transactions: {n_rows:,}")
    print(f"Total Features:     {n_cols}")
    print(f"Normal Transactions: {n_normal:,} ({100 - pct_fraud:.4f}%)")
    print(f"Fraud Transactions:  {n_fraud:,} ({pct_fraud:.4f}%)")
    print("Missing Values:     ", df.isnull().sum().sum())
    print("="*50 + "\n")

    # ----------------------------------------------------
    # Plot 1: Class Distribution (Imbalance visualization)
    # ----------------------------------------------------
    print("[*] Generating Plot 1: Class Distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Count plot with log scale
    sns.countplot(x='Class', data=df, ax=axes[0], hue='Class', palette={0: "#3498db", 1: "#e74c3c"}, legend=False)
    axes[0].set_yscale('log')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Normal (0)', 'Fraud (1)'])
    axes[0].set_title("Transaction Count (Log Scale)")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count (log scale)")
    
    # Add count values on top of bars
    for p in axes[0].patches:
        height = p.get_height()
        axes[0].annotate(f'{int(height):,}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 5), textcoords='offset points')

    # Right: Pie Chart
    axes[1].pie([n_normal, n_fraud], labels=['Normal', 'Fraud'], 
                autopct='%1.3f%%', startangle=90, colors=["#3498db", "#e74c3c"],
                explode=(0, 0.2), textprops={'fontsize': 12, 'weight': 'bold'})
    axes[1].set_title("Transaction Percentage Share")
    
    plt.suptitle("Class Imbalance Analysis", y=0.98, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "class_distribution.png"), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # Plot 2: Time and Amount Distributions
    # ----------------------------------------------------
    print("[*] Generating Plot 2: Time & Amount Distributions...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Time feature analysis: Convert time from seconds to hours (24h cycle)
    df['Hour'] = (df['Time'] / 3600) % 24

    # Time distribution (KDE)
    sns.kdeplot(df[df['Class'] == 0]['Hour'], ax=axes[0, 0], label='Normal', color='#3498db', fill=True, alpha=0.3)
    sns.kdeplot(df[df['Class'] == 1]['Hour'], ax=axes[0, 0], label='Fraud', color='#e74c3c', fill=True, alpha=0.3)
    axes[0, 0].set_title("Transaction Density by Hour of Day")
    axes[0, 0].set_xlabel("Hour of Day (0 - 24)")
    axes[0, 0].set_xlim(0, 24)
    axes[0, 0].legend()

    # Time vs Amount scatter for Fraud
    sns.scatterplot(x='Hour', y='Amount', data=df[df['Class'] == 1], ax=axes[0, 1], color='#e74c3c', alpha=0.7, edgecolor='none')
    axes[0, 1].set_title("Fraud Transactions: Amount vs Hour of Day")
    axes[0, 1].set_xlabel("Hour of Day")
    axes[0, 1].set_ylabel("Amount ($)")
    axes[0, 1].set_xlim(0, 24)

    # Amount distribution (KDE)
    # Using log(Amount + 1) because the amount column is highly skewed
    sns.kdeplot(np.log1p(df[df['Class'] == 0]['Amount']), ax=axes[1, 0], label='Normal', color='#3498db', fill=True, alpha=0.3)
    sns.kdeplot(np.log1p(df[df['Class'] == 1]['Amount']), ax=axes[1, 0], label='Fraud', color='#e74c3c', fill=True, alpha=0.3)
    axes[1, 0].set_title("Transaction Density by Log(Amount + 1)")
    axes[1, 0].set_xlabel("Log(Amount + 1)")
    axes[1, 0].legend()

    # Amount boxplot (with log scale for visual comparison)
    sns.boxplot(x='Class', y='Amount', data=df, ax=axes[1, 1], hue='Class', palette={0: "#3498db", 1: "#e74c3c"}, legend=False)
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_xticklabels(['Normal (0)', 'Fraud (1)'])
    axes[1, 1].set_title("Transaction Amount Boxplot (Log Scale)")
    axes[1, 1].set_xlabel("Class")
    axes[1, 1].set_ylabel("Amount ($) - Log Scale")

    plt.suptitle("Time & Amount Characteristics", y=0.98, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "time_amount_distribution.png"), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # Plot 3: Feature Correlation Heatmap
    # ----------------------------------------------------
    print("[*] Generating Plot 3: Correlation Matrix...")
    # Drop Hour temp feature for correlation matrix
    corr_df = df.drop(columns=['Hour'])
    corr_matrix = corr_df.corr()

    plt.figure(figsize=(18, 14))
    # We use a diverging colormap since correlations range from -1 to 1
    sns.heatmap(corr_matrix, cmap="coolwarm", robust=True, annot=False, fmt=".2f", 
                linewidths=.5, cbar_kws={'shrink': .8}, vmin=-1, vmax=1)
    plt.title("Correlation Matrix of All Features", fontsize=18, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "correlation_matrix.png"), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # Plot 4: Top Correlated V-Features Boxplots
    # ----------------------------------------------------
    print("[*] Generating Plot 4: Key Discriminative V-Features...")
    # Find correlations of features with Class (excluding Class itself and Time/Amount/Hour)
    v_corrs = corr_matrix['Class'].drop(['Class', 'Time', 'Amount'])
    
    # Sort correlations
    sorted_corrs = v_corrs.sort_values()
    
    # Top 4 negative and Top 4 positive correlated V-features
    top_neg = sorted_corrs.head(4)
    top_pos = sorted_corrs.tail(4)
    
    selected_features = list(top_neg.index) + list(top_pos.index)
    
    print("Top negatively correlated features with Class:")
    for feat, val in top_neg.items():
        print(f"  {feat}: {val:.4f}")
    print("Top positively correlated features with Class:")
    for feat, val in top_pos.items():
        print(f"  {feat}: {val:.4f}")

    # Plot subplots for these 8 features
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, feature in enumerate(selected_features):
        sns.boxplot(x='Class', y=feature, data=df, ax=axes[i], hue='Class', palette={0: "#3498db", 1: "#e74c3c"}, legend=False)
        axes[i].set_title(f"{feature} vs Class\n(r = {v_corrs[feature]:.3f})")
        axes[i].set_xlabel("")
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(['Normal (0)', 'Fraud (1)'])
        
        # Remove outlier visualization points or keep them faint to focus on IQR comparison
        # We can also plot distribution instead, but boxplot is great to see the separation
        
    plt.suptitle("Boxplots of Features Most Correlated with Class", y=0.98, fontweight='bold', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "discriminative_features.png"), dpi=300)
    plt.close()

    print("\n[+] All plots successfully generated and saved to the 'plots/' directory:")
    print("  1. plots/class_distribution.png")
    print("  2. plots/time_amount_distribution.png")
    print("  3. plots/correlation_matrix.png")
    print("  4. plots/discriminative_features.png")
    print("="*50)

if __name__ == "__main__":
    main()
