from decision_tree import *
import numpy as np


COL_WIDTH = 13
SEP = " | "


def print_row(*args) -> None:
    print(SEP.join([("{:%ds}" % (COL_WIDTH)).format(str(data))
          for data in args]))


def print_title(title: str, num_cols: int) -> None:
    ROW_WIDTH = COL_WIDTH * num_cols + (num_cols-1) * len(SEP)
    total_len_bar = ROW_WIDTH - 2 - len(title)
    left_len = total_len_bar // 2
    right_len = total_len_bar - left_len
    print(' '.join(["=" * left_len, title, "=" * right_len]))


def print_sep(num_cols: int) -> None:
    print("-+-".join(["-" * (COL_WIDTH)] * num_cols))


def cross_validate(dataset: np.ndarray, prune: bool = False):
    """
    Runs a 10-fold cross validation on the provided dataset, generating 10 trees.

    :return: the metrics and depths of trained trees.
             the metrics are the average of evaluation from all trees.
    """
    shuffled_dataset = np.random.default_rng(seed=1024).permutation(dataset)
    ten_folds = np.array_split(shuffled_dataset, 10)
    metrics = []
    depths = []

    for test_index in range(10):
        trees = []

        test_db = ten_folds[test_index]
        train_db_folds = ten_folds[:test_index] + ten_folds[test_index+1:]

        if prune:
            for valid_index in range(9):
                validation_data = train_db_folds[valid_index]
                training_data = np.concatenate(
                    train_db_folds[:valid_index] + train_db_folds[valid_index+1:])
                tree, _ = decision_tree_learning(training_data, 0)
                tree, depth = pruning(tree, validation_data, training_data)
                trees.append(tree)
                depths.append(depth)
        else:
            training_data = np.concatenate(train_db_folds)
            tree, depth = decision_tree_learning(training_data, 0)
            trees.append(tree)
            depths.append(depth)

        for tree in trees:
            predictions = np.array([execute(data, tree)
                                   for data in test_db], dtype=float)
            metrics.append(generate_evaluation(predictions, test_db[:, -1]))
    metrics = np.swapaxes(np.array(metrics, dtype=object), 0, 1)
    description = ["accuracy", "confusion", "precision", "recall", "f1"]
    return dict(zip(description, metrics)), depths


def main(data_path: str) -> None:
    data = load_data(data_path)
    a_metrics, a_depth = cross_validate(data)
    print("===Pruning Off===")
    print(f"Mean accuracy is: {round(mean(a_metrics['accuracy']), 4)}")
    print(f"Mean depth is: {round(mean(a_depth), 1)}")

    print("===Pruning On===")
    b_metrics, b_depth = cross_validate(data, prune=True)
    print(f"Mean accuracy is: {round(mean(b_metrics['accuracy']), 4)}")
    print(f"Mean depth is: {round(mean(b_depth), 1)}")

    # # The following part renders a tree diagram for the training result
    # # Uncomment to try it out
    # tree, _ = decision_tree_learning(data, 0)
    # create_plot(tree)


if __name__ == '__main__':
    data_path = input('Enter the path of the data set: \n')
    if len(data_path) == 0:
        print("\nTraining ./wifi_db/clean_dataset.txt")
        main("./wifi_db/clean_dataset.txt")
        print("\nTraining ./wifi_db/noisy_dataset.txt")
        main("./wifi_db/noisy_dataset.txt")
    else:
        main(data_path)
