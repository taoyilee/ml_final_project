from homework4.problem_2 import plot_linear

if __name__ == "__main__":
    plot_linear(f"learners/training_auc.pickle", f"learners/validating_auc.pickle", name="2a")
    plot_linear(f"learners/training_auc_2b.pickle", f"learners/validating_auc_2b.pickle", name="2b")
