import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

X = np.genfromtxt('data/X_train.txt', delimiter=None)
Y = np.genfromtxt('data/Y_train.txt', delimiter=None)
scaler = StandardScaler()
X = scaler.fit_transform(X)
pca = PCA(n_components=6)
x_pca = pca.fit_transform(X)
print(x_pca.shape)

x_pca_0 = x_pca[Y == 0, :]
x_pca_1 = x_pca[Y == 1, :]
plt.figure(figsize=(10, 10))
for i in range(5):
    for j in range(i + 1, 6):
        plt.subplot(5, 5, i * 5 + j)
        plt.scatter(x_pca_0[:, i], x_pca_0[:, j], color="red", s=1)
        plt.scatter(x_pca_1[:, i], x_pca_1[:, j], color="blue", s=1)
        plt.xlabel(f"PCA {i+1}")
        plt.ylabel(f"PCA {j+1}")

plt.savefig(f"plot/eda_pca.png")
