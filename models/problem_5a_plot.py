from homework4.problem_5_lib import plot_dnn_mat
import pandas as pd
import numpy as np

df = pd.read_csv("problem_5_summary.csv")
print(df.loc[:, ["hidden_layers", "nodes_per_hidden", "last_dev_auc"]])
print(df.keys())
dnn_hidden = np.array(df["hidden_layers"], dtype=int)
dnn_nodes = np.array(df["nodes_per_hidden"], dtype=int)
last_trn_auc = np.array(df["last_trn_auc"])
last_dev_auc = np.array(df["last_dev_auc"])

dnn_hidden = np.unique(dnn_hidden)
dnn_nodes = np.unique(dnn_nodes)
print(dnn_hidden, dnn_nodes)
dev_auc_mat = last_dev_auc.reshape((len(dnn_hidden), len(dnn_nodes)))
trn_auc_mat = last_trn_auc.reshape((len(dnn_hidden), len(dnn_nodes)))
for i in range(dev_auc_mat.shape[0]):
    for j in range(dev_auc_mat.shape[1]):
        print(f"{dnn_hidden[i]} {dnn_nodes[j]} {dev_auc_mat[i, j]}")

plot_dnn_mat(dnn_nodes, dnn_hidden, trn_auc_mat, dev_auc_mat)
