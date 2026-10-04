"""Fixed conventional feature baselines; no new detector novelty is claimed."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class ForestScore:
    def __init__(self, model):
        self.model = model

    def decision_function(self, values):
        return self.model.predict_proba(values)[:, 1]


class DistanceScore:
    def __init__(self, model):
        self.model = model

    def decision_function(self, values):
        return self.model.mahalanobis(values)


def fit_detector(kind, values, labels, seed, c=0.1):
    if kind == "logistic":
        return LogisticRegression(C=c, class_weight="balanced", max_iter=2000,
                                  random_state=seed).fit(values, labels)
    if kind == "forest":
        return ForestScore(RandomForestClassifier(
            n_estimators=200, min_samples_leaf=10, max_features="sqrt",
            class_weight="balanced", random_state=seed, n_jobs=2,
        ).fit(values, labels))
    if kind == "mahalanobis":
        return DistanceScore(LedoitWolf().fit(values[np.asarray(labels) == 0]))
    raise ValueError("unknown feature detector")
