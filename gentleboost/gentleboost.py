from __future__ import annotations

import collections
import math

from river import base, ensemble


class GentleBoostClassifier(ensemble.AdaBoostClassifier):
    def __init__(
        self,
        model: base.Classifier,
        n_models=10,
        alpha: float = 0.1,
        seed: int | None = None,
    ):
        super().__init__(model, n_models, seed)
        self.alpha = alpha

    def learn_one(self, x, y):
        weight = 1

        for _, model in enumerate(self):
            model.learn_one(x, y, sample_weight=weight)

            if model.predict_one(x) == y:
                weight = weight * (1 + self.alpha)
            else:
                weight = weight * (1 / (1 + self.alpha))

            # clamp weights
            weight = min(max(weight, math.exp(-1)), math.exp(1))
        return self

    def predict_proba_one(self, x, **kwargs):
        y_pred = collections.Counter()
        for classifier in self:
            y_pred.update(classifier.predict_proba_one(x, **kwargs))

        total = sum(y_pred.values())
        if total > 0:
            return {label: proba / total for label, proba in y_pred.items()}
        return y_pred
