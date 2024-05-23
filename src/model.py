import numpy as np
from tqdm import tqdm
from kd_tree import KDNode, Point


class DecisionTree:
    def __init__(self, X, y, max_depth=5):
        self.X, self.y = X, y
        self.tree = [None for _ in range(2 ** (max_depth + 1))]
        self.depth = max_depth

    def train(self):
        self.partition(self.X, self.y, 1)

    def predict(self, X):
        pred_y = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            v = 1
            while not self.tree[v][0]:
                _, j, c = self.tree[v]
                v = 2 * v if X[i, j] != c else 2 * v + 1
            # print(v)
            pred_y[i] = self.tree[v][1]
        return pred_y

    def partition(self, pX, py, idx):
        if idx >= len(self.tree) // 2:
            self.tree[idx] = (True, np.mean(py == 1))
            return

        M, N = pX.shape
        min_loss = float('inf')
        min_j = -1
        min_c = -1
        for j in range(1, N):
            pXj = pX[:, j]
            for c in np.unique(pXj):
                pXj_l, py_l = pXj[pXj != c], py[pXj != c]
                pXj_r, py_r = pXj[pXj == c], py[pXj == c]
                if py_l.size == 0 or py_r.size == 0:
                    continue

                G_l = 1 - (np.mean(py_l == 1)) ** 2 - (np.mean(py_l == -1))
                G_r = 1 - (np.mean(py_r == 1)) ** 2 - (np.mean(py_r == -1))
                G = G_l * np.mean(pXj != c) + G_r * np.mean(pXj == c)

                if G < min_loss:
                    min_loss = G
                    min_j = j
                    min_c = c

        if min_j != -1:
            self.tree[idx] = (False, min_j, min_c)
            pXj = pX[:, min_j]
            self.partition(pX[pXj != min_c], py[pXj != min_c], 2 * idx)
            self.partition(pX[pXj == min_c], py[pXj == min_c], 2 * idx + 1)
        else:
            self.tree[idx] = (True, np.mean(py == 1))


class SVM:
    def __init__(self, X, y, C):
        self.X = X
        self.y = y
        self.C = C
        N, M = X.shape
        self.w = np.zeros((M, 1))

    def train(self, epochs=100, step_size=0.001):
        N, M = self.X.shape

        weights = []
        for epoch in tqdm(range(epochs)):
            for i in range(0, N):
                err = self.y[i] * np.dot(self.X[i, :], self.w)
                if err >= 1:
                    self.w[1:] -= step_size * 2 * self.C * self.w[1:]
                else:
                    self.w[1:] -= step_size * (2 * self.C * self.w[1:] - self.X[i, 1:].reshape(M-1, 1) * self.y[i])
                    self.w[0] += step_size * self.y[i]
            weights += [self.w.copy()]
        return weights

    def predict(self, X):
        y_pred = np.dot(X, self.w)
        return np.where(y_pred > 0, 1, -1)


class KNN:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train(self):
        pass

    def predict(self, X, k=10):
        y_pred = np.zeros((X.shape[0], 1))
        for i in tqdm(range(X.shape[0])):
            dist = np.zeros(self.X.shape[0])
            for j in range(self.X.shape[0]):
                dist[j] = np.linalg.norm(X[i, :] - self.X[j, :])
            closest = np.argsort(dist, axis=0)[:k]
            # print(self.X[closest, :])
            classes = self.y[closest, 0]
            y_pred[i, 0] = np.mean(classes == 1)
            # print(np.sum(classes == 1), np.sum(classes == -1))
        return y_pred
        # dist = np.sum((X-self.X)**2, axis=0)
        # print(dist.shape)
        # y_pred = np.dot(X, self.w)
        # return np.where(y_pred > 0, 1, -1)


class KNN_kdtree:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train(self):
        N, M = self.X.shape
        self.tree = KDNode([(0, 1) for _ in range(M)], capacity=100)
        # print(self.tree.boundaries)
        for i in range(N):
            # print(f'insert: {i}')
            self.tree.insert(Point(self.X[i, :], self.y[i, 0]))

    def predict(self, X, k=10):
        y_pred = np.zeros((X.shape[0], 1))
        for i in tqdm(range(X.shape[0])):
            closest = self.tree.find_closest(X[i, :], k)
            # print(*[p.p for p in closest])
            c1 = len(list(filter(lambda p: p.p.c == 1, closest)))
            y_pred[i, 0] = c1 / len(closest)
            # print(np.sum(classes == 1), np.sum(classes == -1))
        return y_pred
        # dist = np.sum((X-self.X)**2, axis=0)
        # print(dist.shape)
        # y_pred = np.dot(X, self.w)
        # return np.where(y_pred > 0, 1, -1)
