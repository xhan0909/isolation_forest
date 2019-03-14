# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
from scipy import stats
import pandas as pd
import numpy as np
import multiprocessing as mp
from itertools import product
from sklearn.metrics import confusion_matrix


class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees

    def fit(self, X: np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        # if improved:
        #     high_std_idx = np.where(
        #             np.std(X, axis=0) < np.quantile(np.std(X, axis=0), 0.80))[0]
        #     X = X[:, high_std_idx]
            # print(X.shape)
        self.trees = []
        max_height = np.ceil(np.log2(self.sample_size))
        for i in range(self.n_trees):
            X_sample = X[np.random.randint(X.shape[0], size=self.sample_size), :]
            iTree = IsolationTree(0, max_height)
            iTree.fit(X_sample, improved)
            self.trees.append(iTree)

        return self

    def avg_path_length(self, subsample_size):
        if subsample_size > 2:
            harmonic_num = np.log(subsample_size - 1) + 0.5772156649
            avg_path_len = 2 * harmonic_num - 2 * (
                        subsample_size - 1) / subsample_size
        elif subsample_size == 2:
            avg_path_len = 1
        else:
            avg_path_len = 0

        return avg_path_len

    def tree_path_length(self, x, tree):
        "Calculate path length of one instance in one tree."
        tree = tree.root
        while not isinstance(tree, externalNode):
            split_attr_idx = tree.splitAttr
            if x[split_attr_idx] < tree.splitVal:
                tree = tree.left
            else:
                tree = tree.right
        path_len = (tree.current_tree_height - 1) \
                   + self.avg_path_length(tree.size)
        return path_len

    # def path_length(self, X: np.ndarray) -> np.ndarray:
    #     """
    #     Given a 2D matrix of observations, X, compute the average path length
    #     for each observation in X.  Compute the path length for x_i using every
    #     tree in self.trees then compute the average for each x_i.  Return an
    #     ndarray of shape (len(X),1).
    #     """
    #     path_lengths = np.zeros(X.shape[0])
    #     for i, x in enumerate(X):
    #         total_len_per_x = 0
    #         for tree in self.trees:
    #             total_len_per_x += self.tree_path_length(x, tree)
    #         path_lengths[i] = total_len_per_x / self.n_trees
    #
    #     return path_lengths

    def path_length(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        with mp.Pool(processes=4) as pool:
            path_len = pool.starmap(self.tree_path_length,
                                    product(X, self.trees))
        path_lengths = np.array(path_len).reshape(-1, self.n_trees).mean(axis=-1)

        return path_lengths

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        x_scores = np.zeros(X.shape[0])
        scores = self.path_length(X)
        for i, len in enumerate(scores):
            x_scores[i] = 2 ** (-1 * (len / self.avg_path_length(self.sample_size)))

        return x_scores

    def predict_from_anomaly_scores(self, scores: np.ndarray,
                                    threshold: float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        pred = np.array([1 if score >= threshold else 0 for score in scores])
        return pred

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        scores = self.anomaly_score(X)
        predictions = self.predict_from_anomaly_scores(scores, threshold)

        return predictions


class externalNode:
    def __init__(self, size, current_tree_height):
        self.size = size
        self.current_tree_height = current_tree_height


class internalNode:
    def __init__(self, left, right, splitAttr, splitValue, current_tree_height):
        self.left = left
        self.right = right
        self.splitAttr = splitAttr
        self.splitVal = splitValue
        self.current_tree_height = current_tree_height


class IsolationTree:
    def __init__(self, current_height, height_limit, n_nodes=0):
        self.current_height = current_height
        self.height_limit = height_limit
        self.n_nodes = n_nodes

    def fit(self, X: np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        self.root = self.create_child(X, self.current_height, improved)
        self.current_height += 1
        return self.root

    def make_split(self, X, improved):
        if improved:
            rand_attr_idx = np.random.randint(0, X.shape[1], 3)
            split_values = []
            for i in rand_attr_idx:
                feature = X[:, i]
                min_attr = feature.min()
                max_attr = feature.max()
                split_value = np.random.uniform(min_attr, max_attr)
                split_values.append(split_value)
            return rand_attr_idx, split_values

        rand_attr_idx = np.random.randint(0, X.shape[1], 1)[0]
        feature = X[:, rand_attr_idx]
        min_attr = feature.min()
        max_attr = feature.max()
        split_value = np.random.uniform(min_attr, max_attr)
        return rand_attr_idx, split_value

    def create_child(self, X, current_height, improved):
        if current_height >= self.height_limit or X.shape[0] <= 1:
            self.n_nodes += 1
            return externalNode(X.shape[0], current_height)
        else:
            rand_attr_idx, split_value = self.make_split(X, improved)
            if improved:
                # col_mean = X[:, rand_attr_idx].mean(axis=0)
                pos = []
                for i, index in enumerate(rand_attr_idx):
                    pct = stats.percentileofscore(X[:, index], split_value[i])
                    pos.append(np.abs(pct-50))
                rand_attr_idx = rand_attr_idx[np.argmax(pos)]
                split_value = split_value[np.argmax(pos)]
            feature = X[:, rand_attr_idx]
            left_index = feature < split_value
            X_left = X[left_index]
            X_right = X[np.invert(left_index)]
            self.n_nodes += 1

            return internalNode(
                self.create_child(X_left, current_height + 1, improved),
                self.create_child(X_right, current_height + 1, improved),
                rand_attr_idx,
                split_value,
                current_height)


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    thresholds = list(np.arange(1.0, 0, -0.01))
    for thresh in thresholds:
        pred = np.array([1 if score >= thresh else 0 for score in scores])
        confusion = confusion_matrix(y, pred)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)

        if TPR >= desired_TPR:
            FPR = FP / (FP + TN)
            # print(FPR)
            return thresh, FPR
