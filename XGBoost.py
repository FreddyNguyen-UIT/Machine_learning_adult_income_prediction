import numpy as np
from numba import jit, prange
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import pandas as pd

# Standalone sigmoid function for Numba
@jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Histogram-based split finding with Numba
@jit(nopython=True, parallel=True)
def find_histogram_split(x, gradient, hessian, min_leaf, min_child_weight, lambda_, gamma, max_bins=256):
    n = len(x)
    if n < 2 * min_leaf:
        return -np.inf, 0, 0
    
    # Bin feature values
    x_min, x_max = x.min(), x.max()
    if x_min == x_max:
        return -np.inf, 0, 0
    
    bin_edges = np.linspace(x_min, x_max, max_bins + 1)
    hist_gradient = np.zeros(max_bins)
    hist_hessian = np.zeros(max_bins)
    
    # Compute histograms
    for i in prange(n):
        bin_idx = min(int((x[i] - x_min) / (x_max - x_min + 1e-10) * max_bins), max_bins - 1)
        hist_gradient[bin_idx] += gradient[i]
        hist_hessian[bin_idx] += hessian[i]
    
    # Evaluate splits
    best_score = -np.inf
    best_split = 0
    left_gradient = 0
    left_hessian = 0
    total_gradient = hist_gradient.sum()
    total_hessian = hist_hessian.sum()
    
    for i in range(max_bins - 1):
        left_gradient += hist_gradient[i]
        left_hessian += hist_hessian[i]
        right_gradient = total_gradient - left_gradient
        right_hessian = total_hessian - left_hessian
        
        if (left_hessian < min_child_weight or right_hessian < min_child_weight or
            left_hessian < min_leaf or right_hessian < min_leaf):
            continue
        
        gain = 0.5 * (
            (left_gradient**2 / (left_hessian + lambda_)) +
            (right_gradient**2 / (right_hessian + lambda_)) -
            ((left_gradient + right_gradient)**2 / (left_hessian + right_hessian + lambda_))
        ) - gamma
        
        if gain > best_score:
            best_score = gain
            best_split = bin_edges[i + 1]
    
    return best_score, best_split, max_bins

class Node:
    def __init__(self, x, gradient, hessian, idxs, subsample_cols=1.0, min_leaf=5, min_child_weight=1, depth=6, lambda_=1, gamma=0):
        self.x = x
        self.gradient = gradient
        self.hessian = hessian
        self.idxs = idxs
        self.depth = depth
        self.min_leaf = min_leaf
        self.lambda_ = lambda_
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        self.subsample_cols = subsample_cols
        self.column_subsample = np.random.permutation(self.col_count)[:round(self.subsample_cols * self.col_count)]
        
        self.val = -np.sum(gradient[idxs]) / (np.sum(hessian[idxs]) + lambda_)
        self.score = -np.inf
        self.var_idx = 0
        self.split = 0
        self.find_varsplit()
    
    def find_varsplit(self):
        # Parallelize feature evaluation with Numba
        scores = np.full(self.col_count, -np.inf)
        splits = np.zeros(self.col_count)
        for c in self.column_subsample:
            x_col = self.x[self.idxs, c]
            score, split, _ = find_histogram_split(
                x_col, self.gradient[self.idxs], self.hessian[self.idxs],
                self.min_leaf, self.min_child_weight, self.lambda_, self.gamma
            )
            scores[c] = score
            splits[c] = split
        
        best_idx = np.argmax(scores)
        self.score = scores[best_idx]
        self.var_idx = best_idx
        self.split = splits[best_idx]
        
        if self.is_leaf:
            return
        
        x = self.x[self.idxs, self.var_idx]
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        self.lhs = Node(
            self.x, self.gradient, self.hessian, self.idxs[lhs], self.subsample_cols,
            self.min_leaf, self.min_child_weight, self.depth - 1, self.lambda_, self.gamma
        )
        self.rhs = Node(
            self.x, self.gradient, self.hessian, self.idxs[rhs], self.subsample_cols,
            self.min_leaf, self.min_child_weight, self.depth - 1, self.lambda_, self.gamma
        )
    
    @property
    def split_col(self):
        return self.x[self.idxs, self.var_idx]
    
    @property
    def is_leaf(self):
        return self.score == -np.inf or self.depth <= 0
    
    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])
    
    def predict_row(self, xi):
        if self.is_leaf:
            return self.val
        node = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return node.predict_row(xi)

class XGBoostTree:
    def fit(self, x, gradient, hessian, subsample_cols=1.0, min_leaf=5, min_child_weight=1, depth=6, lambda_=1, gamma=0):
        self.dtree = Node(
            x, gradient, hessian, np.arange(len(x)), subsample_cols, min_leaf,
            min_child_weight, depth, lambda_, gamma
        )
        return self
    
    def predict(self, X):
        return self.dtree.predict(X)

class XGBoostClassifier:
    def __init__(self):
        self.estimators = []
    
    @staticmethod
    @jit(nopython=True)
    def grad(preds, labels):
        preds = sigmoid(preds)
        return preds - labels
    
    @staticmethod
    @jit(nopython=True)
    def hess(preds, labels):
        preds = sigmoid(preds)
        return preds * (1 - preds)
    
    def fit(self, X, y, X_val=None, y_val=None, subsample_cols=1.0, min_child_weight=1, depth=6, min_leaf=5, learning_rate=0.05, boosting_rounds=500, lambda_=1, gamma=0, early_stopping_rounds=None, random_state=0):
        np.random.seed(random_state)  # Set seed for reproducibility
        self.X, self.y = X, y
        self.depth = depth
        self.subsample_cols = subsample_cols
        self.min_child_weight = min_child_weight
        self.min_leaf = min_leaf
        self.learning_rate = learning_rate
        self.boosting_rounds = boosting_rounds
        self.lambda_ = lambda_
        self.gamma = gamma
        self.early_stopping_rounds = early_stopping_rounds
        
        # Initialize base prediction (base_score=0.5 corresponds to logit 0)
        self.base_pred = np.full((X.shape[0],), 0.0)
        self.estimators = []
        
        # Early stopping setup
        best_loss = np.inf
        best_round = 0
        if X_val is not None and y_val is not None and early_stopping_rounds:
            val_pred = np.full((X_val.shape[0],), 0.0)
        
        # Sequential tree construction
        for i in range(self.boosting_rounds):
            Grad = self.grad(self.base_pred, y)
            Hess = self.hess(self.base_pred, y)
            tree = XGBoostTree().fit(
                X, Grad, Hess, depth=depth, min_leaf=min_leaf, lambda_=lambda_,
                gamma=gamma, min_child_weight=min_child_weight, subsample_cols=subsample_cols
            )
            self.base_pred += self.learning_rate * tree.predict(X)
            self.estimators.append(tree)
            
            # Early stopping check
            if X_val is not None and y_val is not None and early_stopping_rounds:
                val_pred += self.learning_rate * tree.predict(X_val)
                val_proba = sigmoid(val_pred)
                current_loss = log_loss(y_val, val_proba)
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_round = i
                elif i - best_round >= early_stopping_rounds:
                    print(f"Early stopping at round {i+1}, best loss: {best_loss:.4f}")
                    self.estimators = self.estimators[:best_round + 1]
                    break
    
    def predict_proba(self, X):
        pred = np.zeros(X.shape[0])
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X)
        return sigmoid(np.full((X.shape[0],), 0.0) + pred)
    
    def predict(self, X):
        pred = np.zeros(X.shape[0])
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X)
        predicted_probas = sigmoid(np.full((X.shape[0],), 0.0) + pred)
        return np.where(predicted_probas > 0.5, 1, 0)  # Fixed threshold for binary classification

