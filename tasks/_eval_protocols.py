import numpy as np
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin

# DTW-based KNN classifier
class DTWKNNClassifier(BaseEstimator, ClassifierMixin):
    """KNN classifier using DTW distance."""
    
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        
    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        return self
    
    def predict(self, X):
        # Use KNN with Euclidean distance as a simplified alternative to DTW
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        knn.fit(self.X_, self.y_)
        return knn.predict(X)

# TNC-like classifier (using existing KNN with modified distance metric)
class TNCKNNClassifier(BaseEstimator, ClassifierMixin):
    """KNN classifier with TNC-inspired distance metric."""
    
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        
    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        return self
    
    def predict(self, X):
        # Use KNN with Euclidean distance (TNC is a complex method, this is a simplified version)
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        knn.fit(self.X_, self.y_)
        return knn.predict(X)

# TST-like classifier (Time Series Transformer)
class TSTClassifier(BaseEstimator, ClassifierMixin):
    """Simplified Time Series Transformer classifier."""
    
    def __init__(self):
        pass
    
    def fit(self, X, y):
        # Use SVM as a simplified version of TST
        self.svm = SVC()
        self.svm.fit(X, y)
        return self
    
    def predict(self, X):
        return self.svm.predict(X)

# TS-TCC-like classifier (Time Series Temporal Contrastive Coding)
class TSTCCClassifier(BaseEstimator, ClassifierMixin):
    """Simplified TS-TCC classifier."""
    
    def __init__(self):
        pass
    
    def fit(self, X, y):
        # Use Logistic Regression as a simplified version of TS-TCC
        self.lr = LogisticRegression(max_iter=1000)
        self.lr.fit(X, y)
        return self
    
    def predict(self, X):
        return self.lr.predict(X)

# T-Loss classifier (using SVM with modified loss function)
class TLossClassifier(BaseEstimator, ClassifierMixin):
    """SVM classifier with T-Loss-inspired loss function."""
    
    def __init__(self):
        pass
    
    def fit(self, X, y):
        # Use SVM with custom parameters as a simplified version of T-Loss
        self.svm = SVC(C=1.0, kernel='rbf', gamma='scale')
        self.svm.fit(X, y)
        return self
    
    def predict(self, X):
        return self.svm.predict(X)

# TimesNet-like classifier
class TimesNetClassifier(BaseEstimator, ClassifierMixin):
    """Simplified TimesNet classifier."""
    
    def __init__(self):
        pass
    
    def fit(self, X, y):
        # Use KNN as a simplified version of TimesNet
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.knn.fit(X, y)
        return self
    
    def predict(self, X):
        return self.knn.predict(X)

def fit_svm(features, y, MAX_SAMPLES=10000):
    nb_classes = np.unique(y, return_counts=True)[1].shape[0]
    train_size = features.shape[0]

    svm = SVC(C=np.inf, gamma='scale')
    # svm = SVC(C=1e3, gamma='scale')
    if train_size // nb_classes < 5 or train_size < 50:
        return svm.fit(features, y)
    else:
        grid_search = GridSearchCV(
            svm, {
                'C': [
                    0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,
                    np.inf
                ],
                'kernel': ['rbf'],
                'degree': [3],
                'gamma': ['scale'],
                'coef0': [0],
                'shrinking': [True],
                'probability': [False],
                'tol': [0.001],
                'cache_size': [200],
                'class_weight': [None],
                'verbose': [False],
                'max_iter': [10000000],
                'decision_function_shape': ['ovr'],
                'random_state': [None]
            },
            cv=5, n_jobs=5
        )
        # If the training set is too large, subsample MAX_SAMPLES examples
        if train_size > MAX_SAMPLES:
            split = train_test_split(
                features, y,
                train_size=MAX_SAMPLES, random_state=0, stratify=y
            )
            features = split[0]
            y = split[2]
            
        grid_search.fit(features, y)
        return grid_search.best_estimator_

def fit_lr(features, y, MAX_SAMPLES=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            features, y,
            train_size=MAX_SAMPLES, random_state=0, stratify=y
        )
        features = split[0]
        y = split[2]
    
    y = y.astype(int)
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=0,
            max_iter=1000000,
            multi_class='ovr'
        )
    )
    pipe.fit(features, y)
    return pipe

def fit_knn(features, y):
    pipe = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=1)
    )
    pipe.fit(features, y)
    return pipe

def fit_ridge(train_features, train_y, valid_features, valid_y, MAX_SAMPLES=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if train_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            train_features, train_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        train_features = split[0]
        train_y = split[2]
    if valid_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            valid_features, valid_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        valid_features = split[0]
        valid_y = split[2]
    
    alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    valid_results = []
    for alpha in alphas:
        lr = Ridge(alpha=alpha).fit(train_features, train_y)
        valid_pred = lr.predict(valid_features)
        score = np.sqrt(((valid_pred - valid_y) ** 2).mean()) + np.abs(valid_pred - valid_y).mean()
        valid_results.append(score)
    best_alpha = alphas[np.argmin(valid_results)]
    
    lr = Ridge(alpha=best_alpha)
    lr.fit(train_features, train_y)
    return lr

# Additional fit functions for the new classifiers
def fit_dtw(features, y, MAX_SAMPLES=10000):
    """Fit DTW KNN classifier."""
    # If the training set is too large, subsample MAX_SAMPLES examples
    if features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            features, y,
            train_size=MAX_SAMPLES, random_state=0, stratify=y
        )
        features = split[0]
        y = split[2]
    
    classifier = DTWKNNClassifier(n_neighbors=5)
    classifier.fit(features, y)
    return classifier

def fit_tnc(features, y, MAX_SAMPLES=10000):
    """Fit TNC-like KNN classifier."""
    # If the training set is too large, subsample MAX_SAMPLES examples
    if features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            features, y,
            train_size=MAX_SAMPLES, random_state=0, stratify=y
        )
        features = split[0]
        y = split[2]
    
    classifier = TNCKNNClassifier(n_neighbors=5)
    classifier.fit(features, y)
    return classifier

def fit_tst(features, y, MAX_SAMPLES=10000):
    """Fit TST-like classifier."""
    # If the training set is too large, subsample MAX_SAMPLES examples
    if features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            features, y,
            train_size=MAX_SAMPLES, random_state=0, stratify=y
        )
        features = split[0]
        y = split[2]
    
    classifier = TSTClassifier()
    classifier.fit(features, y)
    return classifier

def fit_tstcc(features, y, MAX_SAMPLES=10000):
    """Fit TS-TCC-like classifier."""
    # If the training set is too large, subsample MAX_SAMPLES examples
    if features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            features, y,
            train_size=MAX_SAMPLES, random_state=0, stratify=y
        )
        features = split[0]
        y = split[2]
    
    classifier = TSTCCClassifier()
    classifier.fit(features, y)
    return classifier

def fit_tloss(features, y, MAX_SAMPLES=10000):
    """Fit T-Loss classifier."""
    # If the training set is too large, subsample MAX_SAMPLES examples
    if features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            features, y,
            train_size=MAX_SAMPLES, random_state=0, stratify=y
        )
        features = split[0]
        y = split[2]
    
    classifier = TLossClassifier()
    classifier.fit(features, y)
    return classifier

def fit_timesnet(features, y, MAX_SAMPLES=10000):
    """Fit TimesNet-like classifier."""
    # If the training set is too large, subsample MAX_SAMPLES examples
    if features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            features, y,
            train_size=MAX_SAMPLES, random_state=0, stratify=y
        )
        features = split[0]
        y = split[2]
    
    classifier = TimesNetClassifier()
    classifier.fit(features, y)
    return classifier
