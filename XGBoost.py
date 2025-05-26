import numpy as np
from numba import jit, prange
from sklearn.metrics import log_loss

@jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

@jit(nopython=True, parallel=True)
def find_histogram_split(x, gradient, hessian, min_leaf, min_child_weight, lambda_, gamma, max_bins=256):
    n = len(x)
    if n < 2 * min_leaf:
        return -np.inf, 0
    
    x_min, x_max = x.min(), x.max()
    if x_min == x_max:
        return -np.inf, 0
    
    bin_width = (x_max - x_min) / max_bins
    hist_gradient = np.zeros(max_bins)
    hist_hessian = np.zeros(max_bins)
    
    for i in prange(n):
        bin_idx = min(int((x[i] - x_min) / bin_width), max_bins - 1)
        hist_gradient[bin_idx] += gradient[i]
        hist_hessian[bin_idx] += hessian[i]
    
    best_score, best_split = -np.inf, 0
    left_grad, left_hess = 0, 0
    total_grad, total_hess = hist_gradient.sum(), hist_hessian.sum()
    
    for i in range(max_bins - 1):
        left_grad += hist_gradient[i]
        left_hess += hist_hessian[i]
        right_grad, right_hess = total_grad - left_grad, total_hess - left_hess
        
        if min(left_hess, right_hess) < max(min_child_weight, min_leaf):
            continue
        
        gain = 0.5 * (left_grad**2 / (left_hess + lambda_) + 
                     right_grad**2 / (right_hess + lambda_) - 
                     total_grad**2 / (total_hess + lambda_)) - gamma
        
        if gain > best_score:
            best_score, best_split = gain, x_min + (i + 1) * bin_width
    
    return best_score, best_split

class Node:
    def __init__(self, x, gradient, hessian, idxs, **params):
        self.x, self.gradient, self.hessian, self.idxs = x, gradient, hessian, idxs
        self.depth = params.get('depth', 6)
        self.min_leaf = params.get('min_leaf', 5)
        self.lambda_ = params.get('lambda_', 1)
        self.gamma = params.get('gamma', 0)
        self.min_child_weight = params.get('min_child_weight', 1)
        self.subsample_cols = params.get('subsample_cols', 1.0)
        
        self.val = -np.sum(gradient[idxs]) / (np.sum(hessian[idxs]) + self.lambda_)
        self.score, self.var_idx, self.split = -np.inf, 0, 0
        
        col_count = x.shape[1]
        self.column_subsample = np.random.choice(col_count, 
                                               int(self.subsample_cols * col_count), 
                                               replace=False)
        self._find_split()
    
    def _find_split(self):
        if self.depth <= 0 or len(self.idxs) < 2 * self.min_leaf:
            return
        
        best_score, best_var, best_split = -np.inf, 0, 0
        
        for c in self.column_subsample:
            score, split = find_histogram_split(
                self.x[self.idxs, c], self.gradient[self.idxs], self.hessian[self.idxs],
                self.min_leaf, self.min_child_weight, self.lambda_, self.gamma
            )
            if score > best_score:
                best_score, best_var, best_split = score, c, split
        
        self.score, self.var_idx, self.split = best_score, best_var, best_split
        
        if self.score > -np.inf:
            mask = self.x[self.idxs, self.var_idx] <= self.split
            lhs_idxs, rhs_idxs = self.idxs[mask], self.idxs[~mask]
            
            params = {'depth': self.depth - 1, 'min_leaf': self.min_leaf,
                     'lambda_': self.lambda_, 'gamma': self.gamma,
                     'min_child_weight': self.min_child_weight,
                     'subsample_cols': self.subsample_cols}
            
            self.lhs = Node(self.x, self.gradient, self.hessian, lhs_idxs, **params)
            self.rhs = Node(self.x, self.gradient, self.hessian, rhs_idxs, **params)
    
    @property
    def is_leaf(self):
        return self.score == -np.inf
    
    def predict(self, X):
        return np.array([self._predict_row(xi) for xi in X])
    
    def _predict_row(self, xi):
        return self.val if self.is_leaf else (
            self.lhs if xi[self.var_idx] <= self.split else self.rhs
        )._predict_row(xi)

class XGBoostTree:
    def fit(self, x, gradient, hessian, **params):
        self.dtree = Node(x, gradient, hessian, np.arange(len(x)), **params)
        return self
    
    def predict(self, X):
        return self.dtree.predict(X)

class XGBoostClassifier:
    def __init__(self):
        self.estimators = []
    
    @staticmethod
    @jit(nopython=True)
    def _grad_hess(preds, labels):
        probs = sigmoid(preds)
        return probs - labels, probs * (1 - probs)
    
    def fit(self, X, y, X_val=None, y_val=None, subsample_cols=1.0, min_child_weight=1, 
            depth=6, min_leaf=5, learning_rate=0.05, boosting_rounds=500, lambda_=1, 
            gamma=0, early_stopping_rounds=None, random_state=0):
        
        np.random.seed(random_state)
        self.learning_rate = learning_rate
        
        pred = np.zeros(X.shape[0])
        val_pred = np.zeros(X_val.shape[0]) if X_val is not None else None
        best_loss, best_round = np.inf, 0
        
        tree_params = {'depth': depth, 'min_leaf': min_leaf, 'lambda_': lambda_,
                      'gamma': gamma, 'min_child_weight': min_child_weight,
                      'subsample_cols': subsample_cols}
        
        for i in range(boosting_rounds):
            grad, hess = self._grad_hess(pred, y)
            tree = XGBoostTree().fit(X, grad, hess, **tree_params)
            pred += learning_rate * tree.predict(X)
            self.estimators.append(tree)
            
            # Early stopping
            if val_pred is not None and early_stopping_rounds:
                val_pred += learning_rate * tree.predict(X_val)
                current_loss = log_loss(y_val, sigmoid(val_pred))
                
                if current_loss < best_loss:
                    best_loss, best_round = current_loss, i
                elif i - best_round >= early_stopping_rounds:
                    print(f"Early stopping at round {i+1}, best loss: {best_loss:.4f}")
                    self.estimators = self.estimators[:best_round + 1]
                    break
        
        return self
    
    def _predict_raw(self, X):
        return sum(self.learning_rate * tree.predict(X) for tree in self.estimators)
    
    def predict_proba(self, X):
        return sigmoid(self._predict_raw(X))
    
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)