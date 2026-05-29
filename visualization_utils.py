import matplotlib.pyplot as plt
import pandas as pd


def plot_accuracy_over_epochs(dense_acc, sparse_keep_ratios, list_of_sparse_accuracies, title="Accuracy over Epochs for Dense and Sparse Models"):
    """
    Plot accuracy curves for dense and sparse models over epochs.
    
    Args:
        dense_acc: list of accuracy values for dense model over epochs
        sparse_keep_ratios: list of keep ratios for sparse models
        list_of_sparse_accuracies: list of (list of accuracies) for each sparse model
        title: title for the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dense_acc, label="Dense (100% Keep Ratio)", marker='o', linewidth=2)
    for keep_ratio, sparse_acc in zip(sparse_keep_ratios, list_of_sparse_accuracies):
        plt.plot(sparse_acc, label=f"Sparse ({keep_ratio*100:.0f}% Keep Ratio)", marker='o', alpha=0.7)
    plt.title(title, fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_training_time_comparison(sparse_keep_ratios, list_of_sparse_times, dense_time, title="Training Time for Dense and Sparse Models"):
    """
    Plot training time comparison as bar chart for sparse models with dense model as reference line.
    
    Args:
        sparse_keep_ratios: list of keep ratios for sparse models
        list_of_sparse_times: list of training times for each sparse model
        dense_time: training time for dense model
        title: title for the plot
    """
    plt.figure(figsize=(12, 6))
    plt.bar(
        [f"{keep_ratio*100:.0f}%" for keep_ratio in sparse_keep_ratios],
        list_of_sparse_times,
        label="Sparse Models",
        color='steelblue',
        alpha=0.7
    )
    plt.axhline(y=dense_time, color='red', linestyle='--', linewidth=2, label=f"Dense Model ({dense_time:.2f}s)")
    plt.title(title, fontsize=14)
    plt.xlabel("Keep Ratio", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def create_training_time_table(sparse_keep_ratios, list_of_sparse_times, dense_time):
    """
    Create and display a table comparing training times.
    
    Args:
        sparse_keep_ratios: list of keep ratios for sparse models
        list_of_sparse_times: list of training times for each sparse model
        dense_time: training time for dense model
        
    Returns:
        DataFrame with training time comparison
    """
    data = {
        "Keep Ratio": [f"{keep_ratio*100:.0f}%" for keep_ratio in sparse_keep_ratios] + ["Dense (100%)"],
        "Training Time (seconds)": list_of_sparse_times + [dense_time]
    }
    df = pd.DataFrame(data)
    return df


def create_accuracy_summary_table(sparse_keep_ratios, list_of_sparse_accuracies, dense_acc):
    """
    Create and display a summary table of final accuracies for all models.
    
    Args:
        sparse_keep_ratios: list of keep ratios for sparse models
        list_of_sparse_accuracies: list of (list of accuracies) for each sparse model
        dense_acc: list of accuracy values for dense model
        
    Returns:
        DataFrame with accuracy summary
    """
    final_accs = [acc[-1] for acc in list_of_sparse_accuracies]
    data = {
        "Keep Ratio": [f"{keep_ratio*100:.0f}%" for keep_ratio in sparse_keep_ratios] + ["Dense (100%)"],
        "Final Accuracy": final_accs + [dense_acc[-1]]
    }
    df = pd.DataFrame(data)
    return df


__all__ = [
    "plot_accuracy_over_epochs",
    "plot_training_time_comparison",
    "create_training_time_table",
    "create_accuracy_summary_table"
]
