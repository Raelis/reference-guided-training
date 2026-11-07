import os
import pandas as pd

all_dfs = []

ROOT = "results"

for folder in os.listdir(ROOT):
    exp_path = os.path.join(ROOT, folder)
    if not os.path.isdir(exp_path):
        continue

    csv_path = os.path.join(exp_path, "experiment-results.csv")
    if not os.path.exists(csv_path):
        continue
    
    df = pd.read_csv(csv_path)

    all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True)

df_all = df_all.sort_values(["model"]).reset_index(drop=True)
df_all.to_csv("processed_results/results.csv", index=False)