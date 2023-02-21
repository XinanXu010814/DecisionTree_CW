from matplotlib import pyplot as plt
import numpy as np

def load_data(data_path):
    return np.loadtxt(data_path)

def calc_info_entropy(label):
    """
    :param label: the list of frequency of distinct labels
    :return: the info entropy, by calculating the probability of each label
    """
    probs = label[label != 0] / sum(label)
    return -sum(probs*(np.log2(probs)))


def find_split(training_dataset):
    labels, y, labels_count = np.unique(
        training_dataset[:, -1], return_inverse=True, return_counts=True)
    h_all = calc_info_entropy(labels_count)
    max_ig = split_attr = split_value = None

    for attr_idx in range(training_dataset.shape[1] - 1):
        indexes = training_dataset[:, attr_idx].argsort()
        dataset = training_dataset[indexes, :]
        y_ = y[indexes]

        num_samples = dataset.shape[0]
        left_labels = np.zeros((len(labels),))
        for sample_idx in range(1, num_samples):
            left_labels[y_[sample_idx-1]] += 1
            # skip on identical values - cannot find a split value
            if dataset[sample_idx, attr_idx] == dataset[sample_idx-1, attr_idx]:
                continue
            # calculate entrophy gain
            remainder = sample_idx / num_samples * calc_info_entropy(left_labels) + (
                num_samples - sample_idx) / num_samples * calc_info_entropy(labels_count - left_labels)
            if (max_ig is None) or (max_ig < h_all - remainder):
                max_ig = h_all - remainder
                split_attr, split_value = attr_idx, (
                    dataset[sample_idx, attr_idx] + dataset[sample_idx-1, attr_idx])/2
    return split_attr, split_value


def decision_tree_learning(training_dataset, depth):
    """
    Trains a overfitted decision tree and returns the root and max_depth.

    :return: (root, maximum depth)
    """
    if np.all(training_dataset[:, -1] == training_dataset[:, -1][0]):
        return ({'leaf': True, 'label': training_dataset[:, -1][0]}, depth)
    else:
        attribute, split_value = find_split(training_dataset)
        node = {'leaf': False, 'attribute': attribute, 'value': split_value}
        l_dataset = training_dataset[training_dataset[:, attribute] < split_value]
        r_dataset = training_dataset[training_dataset[:, attribute] >= split_value]

        l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)
        r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)
        node['l_branch'] = l_branch
        node['r_branch'] = r_branch
        return (node, max(l_depth, r_depth))


def execute(data, tree):
    """
    :return: the predicted label according to the given data and tree
    """
    if tree['leaf']:
        return tree['label']
    elif data[tree['attribute']] < tree['value']:
        return execute(data, tree['l_branch'])
    else:
        return execute(data, tree['r_branch'])

# rows are the actual classes, columns are the predicted classes
def confusion_matrix(preds, correct_labels):
    labels = np.sort(np.unique(np.concatenate((preds, correct_labels))))
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for i, label in enumerate(labels):
        correct_indices = correct_labels == label
        unique_labels, counts = np.unique(preds[correct_indices], return_counts=True)
        freq_dict = dict(zip(unique_labels, counts))
        for j, column_label in enumerate(labels):
            matrix[i, j] = freq_dict.get(column_label, 0)
    return matrix



def accuracy(preds, correct_labels):
    confusion = confusion_matrix(preds, correct_labels)
    if np.sum(confusion) > 0:
        return np.sum(np.diagonal(confusion)) / np.sum(confusion)
    else:
        return 0.


def precision(preds, correct_labels):
    confusion = confusion_matrix(preds, correct_labels)
    ps = np.zeros(len(confusion))
    for i in range(len(confusion)):
        if np.sum(confusion[:, i]) > 0:
            ps[i] = confusion[i, i] / np.sum(confusion[:, i])
    return ps


def recall(preds, correct_labels):
    confusion = confusion_matrix(preds, correct_labels)
    rs = np.zeros(len(confusion))
    for i in range(len(confusion)):
        if np.sum(confusion[i, :]) > 0:
            rs[i] = confusion[i, i] / np.sum(confusion[i, :])
    return rs


def f1(preds, correct_labels):
    precisions = precision(preds, correct_labels)
    recalls = recall(preds, correct_labels)
    f = np.zeros((len(precisions), ))
    for c, (p, r) in enumerate(zip(precisions, recalls)):
        if p + r > 0:
            f[c] = 2 * p * r / (p + r)
    return f


def mean(ints):
    return np.sum(ints, axis=0) / len(ints)

def generate_evaluation(preds, correct_labels):
    mean_accuracy = accuracy(preds, correct_labels)
    confusions = confusion_matrix(preds, correct_labels)
    precisions = precision(preds, correct_labels)
    recalls = recall(preds, correct_labels)
    f1s = f1(preds, correct_labels)
    return (mean_accuracy, confusions, precisions, recalls, f1s)


def evaluate(test_db, trained_tree):
    if len(test_db) == 0:
        return 0
    preds = np.array([execute(data, trained_tree) for data in test_db])
    return len(preds[preds == test_db[:, -1]])/len(test_db)


def pruning(root, validation_dataset, training_dataset) -> int:
    """
    Pruning the tree's over-fitting leaves according to the validation_dataset

    :param tree: the decision tree, type is a dictionary with fields:
                'leaf': bool,
                'label': any,
                'value': number,
                'attribute': number,
                'l_branch': type(this),
                'r_branch': type(this),
    :param validation_dataset: some np array with its last col as label
    :param training_dataset: same format as above
    :return: max depth after pruning
    """
    max_depth = 0

    def pruning_helper(node, depth, validation_dataset, training_dataset) -> None:
        if not node['leaf']:
            l_validation_dataset \
                = validation_dataset[validation_dataset[:, node['attribute']] < node['value']]
            l_training_dataset = training_dataset[training_dataset[:, node['attribute']] < node['value']]
            left = node['l_branch'] = pruning_helper(
                node['l_branch'], depth + 1, l_validation_dataset, l_training_dataset)
            r_validation_dataset \
                = validation_dataset[validation_dataset[:, node['attribute']] >= node['value']]
            r_training_dataset = training_dataset[training_dataset[:, node['attribute']] >= node['value']]
            right = node['r_branch'] = pruning_helper(
                node['r_branch'], depth + 1, r_validation_dataset, r_training_dataset)
            if left['leaf'] and right['leaf']:
                original_p = evaluate(validation_dataset, node)
                labels, counts = np.unique(training_dataset[:,-1], return_counts=True)
                majority_class_label = labels[counts.argmax()]
                new_node = {"leaf" : True, "label" : majority_class_label}
                new_p = evaluate(validation_dataset, new_node)
                if new_p >= original_p:
                    return new_node
                else:
                    nonlocal max_depth
                    max_depth = max(max_depth, depth + 1)
        return node

    pruning_helper(root, 0, validation_dataset, training_dataset)
    return (root, max_depth)


def get_height(node):
    if node['leaf']:
        return 1
    else:
        return max(get_height(node['l_branch']), get_height(node['r_branch'])) + 1


def draw_node(node, depth, nodes_by_level):
    if node['leaf']:
        nodes_by_level[depth].append(
            (len(nodes_by_level[depth-1]), 'leaf:{}'.format(node['label'])))
    else:
        draw_node(node['l_branch'], depth + 1, nodes_by_level)
        draw_node(node['r_branch'], depth + 1, nodes_by_level)
        nodes_by_level[depth].append(
            (len(nodes_by_level[depth-1]), 'x' + str(node['attribute']) + ' < ' + str(node['value'])))


def create_plot(root):
    height = get_height(root)
    nodes_by_level = [[] for _ in range(height)]
    draw_node(root, 0, nodes_by_level)

    def annotate(fromx, fromy, x, y, msg, arrow):
        plt.annotate(msg, xy=(fromx, fromy), xytext=(x, y), va='center', ha='center', arrowprops=arrow, xycoords='axes fraction',
                     bbox=dict(boxstyle='round', fc='0.8'), fontsize=12)
    for (index, nodes) in enumerate(nodes_by_level[::-1]):
        level = height - index - 1
        for (count, node) in enumerate(nodes):
            if level == 0:
                annotate(0, 0, 0.5, 1 - 1/(height + 1), nodes_by_level[0][0][1], None)
            else:
                (from_count, msg) = node
                fromx = (from_count + 1) / (len(nodes_by_level[level - 1]) + 1)
                tox = (count + 1) / (len(nodes_by_level[level]) + 1)
                fromy = 1 - (level)/(height + 1)
                toy = 1 - (level + 1) / (height + 1)
                annotate(fromx, fromy, tox, toy, msg, dict(arrowstyle='<-'))

    # remove x, y axis scale labels
    plt.xticks([])
    plt.yticks([])
    plt.autoscale()
    plt.tight_layout()
    plt.show()  # display
