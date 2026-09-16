"""TF-IDF + linear SVM, the classical LexGLUE baseline (word and bigram features, one-vs-rest for multi-label).

``C`` is selected on the development split; decision values are mapped
through a logistic (multi-label) or soft-max (single-label) so that the same
prediction files and metrics can be produced as for the neural models.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from lkhit.data.dataset import Document
from lkhit.data.tasks import TaskSpec

logger = logging.getLogger("lkhit.models.tfidf_svm")


class TfidfSvmClassifier:
    def __init__(self, spec: TaskSpec, model_cfg: dict) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.spec = spec
        self.cfg = model_cfg
        kwargs = dict(
            ngram_range=tuple(model_cfg.get("ngram_range", [1, 2])),
            min_df=int(model_cfg.get("min_df", 5)),
            max_features=model_cfg.get("max_features", 200_000),
            sublinear_tf=True,
            dtype=np.float32,
        )
        if spec.language == "zh":
            from lkhit.data.word_level import tokenize_zh

            kwargs.update(tokenizer=tokenize_zh, token_pattern=None)
        else:
            kwargs.update(lowercase=True, stop_words="english")
        self.vectoriser = TfidfVectorizer(**kwargs)
        self.classifier = None
        self.best_c = None

    @staticmethod
    def texts(documents: Sequence[Document]) -> list[str]:
        return [doc.full_text("\n") for doc in documents]

    def targets(self, documents: Sequence[Document]) -> np.ndarray:
        if self.spec.multi_label:
            y = np.zeros((len(documents), self.spec.n_labels), dtype=np.int64)
            for i, doc in enumerate(documents):
                labels = list(doc.labels) or ([self.spec.none_index] if self.spec.none_index is not None else [])
                y[i, labels] = 1
            return y
        return np.asarray([doc.labels[0] for doc in documents], dtype=np.int64)

    def _make_classifier(self, c: float):
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.svm import LinearSVC

        svm = LinearSVC(C=c, class_weight=self.cfg.get("class_weight"), max_iter=int(self.cfg.get("max_iter", 5000)))
        return OneVsRestClassifier(svm, n_jobs=int(self.cfg.get("n_jobs", -1))) if self.spec.multi_label else svm

    def fit(self, train: Sequence[Document], dev: Sequence[Document]) -> dict:
        from lkhit.metrics import decisions_from_probs, macro_f1, one_hot

        x_train = self.vectoriser.fit_transform(self.texts(train))
        x_dev = self.vectoriser.transform(self.texts(dev))
        y_train, y_dev = self.targets(train), self.targets(dev)
        logger.info("TF-IDF vocabulary: %d features", x_train.shape[1])
        grid = list(self.cfg.get("c_grid", [0.1, 0.5, 1.0, 5.0, 10.0]))
        history = {}
        best_score, best_clf = -1.0, None
        for c in grid:
            clf = self._make_classifier(c).fit(x_train, y_train)
            probs = self._scores_to_probs(clf.decision_function(x_dev))
            score = macro_f1(decisions_from_probs(probs, self.spec), one_hot(y_dev, self.spec.n_labels))
            history[str(c)] = 100 * score
            logger.info("C=%g: dev macro-F1 %.2f", c, 100 * score)
            if score > best_score:
                best_score, best_clf, self.best_c = score, clf, c
        self.classifier = best_clf
        return {"c_grid": history, "best_c": self.best_c, "dev_macro_f1": 100 * best_score, "n_features": int(x_train.shape[1])}

    def _scores_to_probs(self, decision: np.ndarray) -> np.ndarray:
        decision = np.asarray(decision, dtype=np.float64)
        if decision.ndim == 1:
            decision = np.stack([-decision, decision], axis=1)
        if self.spec.multi_label:
            return 1.0 / (1.0 + np.exp(-decision))
        shifted = decision - decision.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict_logits(self, documents: Sequence[Document]) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("fit() must be called first")
        return np.asarray(self.classifier.decision_function(self.vectoriser.transform(self.texts(documents))), dtype=np.float64)

    def predict_proba(self, documents: Sequence[Document]) -> np.ndarray:
        return self._scores_to_probs(self.predict_logits(documents))

    def save(self, path) -> None:
        import joblib

        joblib.dump({"vectoriser": self.vectoriser, "classifier": self.classifier, "best_c": self.best_c, "spec": self.spec.to_dict()}, path)

    @classmethod
    def load(cls, path, model_cfg: dict) -> "TfidfSvmClassifier":
        import joblib

        payload = joblib.load(path)
        obj = cls(TaskSpec.from_dict(payload["spec"]), model_cfg)
        obj.vectoriser, obj.classifier, obj.best_c = payload["vectoriser"], payload["classifier"], payload["best_c"]
        return obj
