import matplotlib.pyplot as plt
import pandas as pd


def _normalize_accuracy_inputs(dense_acc_or_results, sparse_keep_ratios=None, list_of_sparse_accuracies=None):
    if isinstance(dense_acc_or_results, dict):
        dense_acc = dense_acc_or_results["dense"]
        sparse_map = dense_acc_or_results["sparse"]
        keep_ratios = list(sparse_map.keys())
        sparse_accuracies = list(sparse_map.values())
        return dense_acc, keep_ratios, sparse_accuracies

    dense_acc = dense_acc_or_results
    if isinstance(sparse_keep_ratios, dict) and list_of_sparse_accuracies is None:
        keep_ratios = list(sparse_keep_ratios.keys())
        sparse_accuracies = list(sparse_keep_ratios.values())
        return dense_acc, keep_ratios, sparse_accuracies

    if isinstance(list_of_sparse_accuracies, str) and not isinstance(sparse_keep_ratios, str):
        # legacy notebook call: plot_accuracy_over_epochs(dense_acc, sparse_dict, dataset_name)
        keep_ratios = list(sparse_keep_ratios.keys())
        sparse_accuracies = list(sparse_keep_ratios.values())
        return dense_acc, keep_ratios, sparse_accuracies, list_of_sparse_accuracies

    return dense_acc, sparse_keep_ratios, list_of_sparse_accuracies


def _normalize_time_inputs(sparse_keep_ratios_or_results, list_of_sparse_times=None, dense_time=None):
    if isinstance(sparse_keep_ratios_or_results, dict):
        dense_time = sparse_keep_ratios_or_results["dense"]
        sparse_map = sparse_keep_ratios_or_results["sparse"]
        keep_ratios = list(sparse_map.keys())
        sparse_times = list(sparse_map.values())
        return keep_ratios, sparse_times, dense_time

    if isinstance(list_of_sparse_times, dict) and dense_time is None:
        keep_ratios = list(list_of_sparse_times.keys())
        sparse_times = list(list_of_sparse_times.values())
        return keep_ratios, sparse_times, sparse_keep_ratios_or_results

    return sparse_keep_ratios_or_results, list_of_sparse_times, dense_time


def plot_accuracy_over_epochs(dense_acc, sparse_keep_ratios, list_of_sparse_accuracies, title="Accuracy over Epochs for Dense and Sparse Models"):
    """
    Plot accuracy curves for dense and sparse models over epochs.
    
    Args:
        dense_acc: list of accuracy values for dense model over epochs
        sparse_keep_ratios: list of keep ratios for sparse models
        list_of_sparse_accuracies: list of (list of accuracies) for each sparse model
        title: title for the plot
    """
    if isinstance(sparse_keep_ratios, dict) and list_of_sparse_accuracies is None:
        dense_acc, sparse_keep_ratios, list_of_sparse_accuracies = _normalize_accuracy_inputs(dense_acc, sparse_keep_ratios, list_of_sparse_accuracies)
    elif isinstance(dense_acc, dict):
        dense_acc, sparse_keep_ratios, list_of_sparse_accuracies = _normalize_accuracy_inputs(dense_acc)
    elif isinstance(list_of_sparse_accuracies, str):
        dense_acc, sparse_keep_ratios, list_of_sparse_accuracies, title = _normalize_accuracy_inputs(dense_acc, sparse_keep_ratios, list_of_sparse_accuracies)

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
    if isinstance(sparse_keep_ratios, dict) and list_of_sparse_times is None:
        sparse_keep_ratios, list_of_sparse_times, dense_time = _normalize_time_inputs(sparse_keep_ratios)
    elif isinstance(list_of_sparse_times, dict) and dense_time is None:
        sparse_keep_ratios, list_of_sparse_times, dense_time = _normalize_time_inputs(sparse_keep_ratios, list_of_sparse_times, dense_time)
    elif isinstance(dense_time, str):
        title = dense_time
        sparse_keep_ratios, list_of_sparse_times, dense_time = _normalize_time_inputs(sparse_keep_ratios, list_of_sparse_times, None)

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
    if isinstance(sparse_keep_ratios, dict) and list_of_sparse_times is None:
        sparse_keep_ratios, list_of_sparse_times, dense_time = _normalize_time_inputs(sparse_keep_ratios)

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
    if isinstance(sparse_keep_ratios, dict) and list_of_sparse_accuracies is None:
        dense_acc, sparse_keep_ratios, list_of_sparse_accuracies = _normalize_accuracy_inputs(sparse_keep_ratios)

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
