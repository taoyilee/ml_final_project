from homework4.problem_4 import plot_dt, plot_dt_mat, plot_dt_mLeaf
import numpy as np

if __name__ == "__main__":
    k = list(range(1, 50, 2))
    alpha = np.linspace(0, 5, len(k))
    alpha = alpha.round(2)

    maxDepth = list(range(3, 16))
    print(f"maxDepth = {maxDepth}")
    plot_dt(maxDepth, "learners/training_auc_4a.pickle",
            f"learners/validating_auc_4a.pickle",
            name="4a")

    maxDepth = list(range(5, 15))
    print(maxDepth)

    minLeaf = list(range(6, 9))
    print(minLeaf)

    print(f"maxDepth = {maxDepth}")
    print(f"minLeaf = {minLeaf}")
    plot_dt_mLeaf(maxDepth, minLeaf, "learners/training_auc_minLeaf_4b.pickle",
                  f"learners/validating_auc_minLeaf_4b.pickle",
                  name="4b_minLeaf")

    minParent = list(range(2, 20))
    minLeaf = list(range(2, 20))
    for depth_4c in [6, 7, 8]:
        plot_dt_mat(minParent, minLeaf, f"learners/training_auc_minParent_4c_{depth_4c}.pickle",
                    f"learners/validating_auc_minParent_4c_{depth_4c}.pickle",
                    name=f"4c_mDepth_{depth_4c}")
