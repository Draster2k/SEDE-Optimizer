import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def clean_data(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace('[', '', regex=False).str.replace(']', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def plot_all_boxplots():
    print("📊 Generating Box Plots for ALL dimensions...")
    if not os.path.exists("Results"):
        return

    # Look for ALL CSV files now
    files = [f for f in os.listdir("Results") if f.endswith(".csv") and "table" not in f]
    
    for file in files:
        func_name = file.replace(".csv", "").replace("_", " ")
        try:
            df = pd.read_csv(f"Results/{file}")
            df = clean_data(df)
            df_melt = df.melt(var_name="Algorithm", value_name="Fitness")
            df_melt = df_melt[~df_melt["Algorithm"].str.contains("time")]
            
            plt.figure(figsize=(8, 5))
            sns.boxplot(x="Algorithm", y="Fitness", data=df_melt, hue="Algorithm", palette="Set2", legend=False)
            
            # Auto-log scale if the gap between algos is huge
            if (df_melt["Fitness"] > 0).all() and (df_melt["Fitness"].max() / (df_melt["Fitness"].min() + 1e-9) > 100):
                plt.yscale("log")
            
            plt.title(f"Fitness: {func_name}")
            plt.tight_layout()
            
            save_path = f"Results/{file.replace('.csv', '_boxplot.png')}"
            plt.savefig(save_path, dpi=300)
            print(f"   -> Created {save_path}")
            plt.close()
        except Exception as e:
            print(f"   ⚠️ Error plotting {file}: {e}")

if __name__ == "__main__":
    plot_all_boxplots()