from homework4.problem_3 import plot_knn_core, plot_knn_combined_core, plot_knn_alpha, plot_knn_alpha_mat
import numpy as np
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("problem_3.csv")
    non_scaled = df.loc[df["scaled"] == 0].loc[df["trn_p"] == 20.0].loc[df["alpha"] == 0.5]
    scaled = df.loc[df["scaled"] == 1].loc[df["trn_p"] == 20.0].loc[df["alpha"] == 0.5]

    k = np.unique(df[["k"]])
    k.sort()
    k = [ki for ki in k if ki < 40]
    k_rows = scaled.loc[scaled["k"] == k[0]][:3]
    train_auc_scaled = np.array(k_rows[['trn_auc']])
    val_auc_scaled = np.array(k_rows[['dev_auc']])

    for ki in k[1:]:
        k_rows = scaled.loc[scaled["k"] == ki][:3]
        train_auc_scaled = np.concatenate((train_auc_scaled, np.array(k_rows[['trn_auc']])), axis=1)
        val_auc_scaled = np.concatenate((val_auc_scaled, np.array(k_rows[['dev_auc']])), axis=1)
    print(train_auc_scaled.shape)
    plot_knn_core(k, train_auc_scaled, val_auc_scaled, name="3a")

    k_rows = non_scaled.loc[non_scaled["k"] == k[0]][:3]
    train_auc_unscaled = np.array(k_rows[['trn_auc']])
    val_auc_unscaled = np.array(k_rows[['dev_auc']])
    for ki in k[1:]:
        k_rows = non_scaled.loc[non_scaled["k"] == ki][:3]
        train_auc_unscaled = np.concatenate((train_auc_unscaled, np.array(k_rows[['trn_auc']])), axis=1)
        val_auc_unscaled = np.concatenate((val_auc_unscaled, np.array(k_rows[['dev_auc']])), axis=1)
    plot_knn_core(k, train_auc_unscaled, val_auc_unscaled, name="3b")

    plot_knn_combined_core(k, (train_auc_scaled, train_auc_unscaled), (val_auc_scaled, val_auc_unscaled),
                           name=("Rescaled", "Not_Rescaled"))
    alpha = np.unique(df[["alpha"]])
    alpha.sort()
    # print(alpha)
    # plot_knn_alpha(k, alpha, f"learners/training_auc_3c_scaled.pickle", f"learners/validating_auc_3c_scaled.pickle",
    #                name="3c_scaled")
    #
    # k = list(range(1, 50, 2))
    # print(k)
    # alpha = np.linspace(0, 5, len(k))
    # alpha = alpha.round(2)
    # print(alpha)
    # plot_knn_alpha(k, alpha, f"learners/training_auc_3c_unscaled.pickle", f"learners/validating_auc_3c_unscaled.pickle",
    #                name="3c_unscaled")
    #
    # plot_knn_alpha_mat(k, alpha, f"learners/training_auc_3c_scaled.pickle", f"learners/validating_auc_3c_scaled.pickle",
    #                    name="3c_scaled")
    # plot_knn_alpha_mat(k, alpha, f"learners/training_auc_3c_unscaled.pickle",
    #                    f"learners/validating_auc_3c_unscaled.pickle",
    #                    name="3c_unscaled")
