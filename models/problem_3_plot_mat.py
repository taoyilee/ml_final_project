from homework4.problem_3 import plot_knn_core, plot_knn_combined_core, plot_knn_alpha, plot_knn_alpha_mat
import numpy as np
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("problem_3.csv")
    scaled = df.loc[df["scaled"] == 1].loc[df["trn_p"] == 20.0]

    k = np.unique(df[["k"]])
    k.sort()
    alpha = np.unique(df[["alpha"]])
    alpha.sort()
    print(k)
    print(alpha)

    k = [ki for ki in k if ki < 21]
    k_rows = scaled.loc[scaled["k"] == k[0]][:3]
    train_auc_scaled = np.zeros((3, len(alpha), len(k)))
    val_auc_scaled = np.zeros((3, len(alpha), len(k)))

    for i, alphai in enumerate(alpha):
        for j, ki in enumerate(k):
            k_rows = scaled.loc[scaled["k"] == ki].loc[scaled["alpha"] == alphai][:3]
            print(f"{alphai} {ki} {len(k_rows)}")
            train_auc_scaled[:, i, j] = np.array(k_rows[['trn_auc']]).flatten()
            val_auc_scaled[:, i, j] = np.array(k_rows[['dev_auc']]).flatten()

    plot_knn_alpha_mat(k, alpha, f"learners/training_auc_3c_scaled.pickle", f"learners/validating_auc_3c_scaled.pickle",
                       name="3c_scaled")
