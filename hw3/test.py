import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

rng = np.random.default_rng()


def gini(sequence):
    _, c = np.unique(sequence, return_counts=True)
    p = c / sequence.size
    return 1 - np.sum(p ** 2)


def entropy(sequence):
    _, c = np.unique(sequence, return_counts=True)
    p = c / sequence.size
    return -np.sum(p * np.log2(p))


class Criteria:
    @staticmethod
    def gini(sequence):
        return gini(sequence)

    @staticmethod
    def entropy(sequence):
        return entropy(sequence)


def find_best_split(x_data, y_data, criterion):
    if np.unique(y_data).size == 1:
        return None, None, None, None, None, None
    criteria = getattr(Criteria, criterion)
    best_split_index = None
    best_split_target = None
    best_split_purity = 2
    for feature_index, feature in enumerate(x_data.T):
        values = np.unique(feature)
        for val in values:
            mask = feature <= val
            l_y = y_data[mask]
            r_y = y_data[~mask]
            if l_y.size == 0 or r_y.size == 0:
                continue
            purity = (criteria(l_y) * l_y.size + criteria(r_y) * r_y.size) / y_data.size
            if purity <= best_split_purity:
                best_split_purity = purity
                best_split_target = val
                best_split_index = feature_index

    split_mask = x_data.T[best_split_index] <= best_split_target
    l_x = x_data[split_mask]
    r_x = x_data[~split_mask]
    l_y = y_data[split_mask]
    r_y = y_data[~split_mask]

    return l_x, r_x, l_y, r_y, best_split_index, best_split_target


class DecisionTree:
    def __init__(self, criterion='gini', max_depth=None, cur_depth=0):
        self.criterion = criterion
        self.max_depth = max_depth
        self.cur_depth = cur_depth
        self.l_branch = None
        self.r_branch = None
        self.split_index = None
        self.target = None
        self.predict_data = None
        self.purity = 0

    def fit(self, x_data, y_data):
        l_x, r_x, l_y, r_y, split_index, target = find_best_split(x_data, y_data, self.criterion)
        if self.cur_depth == self.max_depth or l_x is None or r_x is None:
            u, c = np.unique(y_data, return_counts=True)
            self.predict_data = u[np.argmax(c)]
            return
        self.l_branch = DecisionTree(criterion=self.criterion, max_depth=self.max_depth, cur_depth=self.cur_depth+1)
        self.r_branch = DecisionTree(criterion=self.criterion, max_depth=self.max_depth, cur_depth=self.cur_depth+1)
        self.split_index = split_index
        self.target = target

        self.l_branch.fit(l_x, l_y)
        self.r_branch.fit(r_x, r_y)

    def predict(self, x_data):
        result = []
        for features in x_data:
            result.append(self.predict_(features))
        return np.asarray(result)

    def predict_(self, features):
        if self.predict_data is not None:
            return self.predict_data
        if features[self.split_index] <= self.target:
            return self.l_branch.predict_(features)
        else:
            return self.r_branch.predict_(features)


def load_dataset():
    file_url = "http://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv"
    df = pd.read_csv(
        file_url,
        names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
               "Viscera weight", "Shell weight", "Age"]
    )

    df['Target'] = (df["Age"] > 15).astype(int)
    df = df.drop(labels=["Age"], axis="columns")

    train_idx = range(0, len(df), 10)
    test_idx = range(1, len(df), 20)

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    x_train = train_df.drop(labels=["Target"], axis="columns")
    feature_names = x_train.columns.values
    x_train = x_train.values
    y_train = train_df['Target'].values

    x_test = test_df.drop(labels=["Target"], axis="columns")
    x_test = x_test.values
    y_test = test_df['Target'].values
    return x_train, y_train, x_test, y_test, feature_names


if __name__ == "__main__":
    x_train, y_train, x_test, y_test, feature_names = load_dataset()

    clf = DecisionTree(criterion='gini', max_depth=3)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print(accuracy_score(y_test, y_pred))
