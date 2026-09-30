"""TF–IDF + linear SVM, under the task's label cardinality."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


def _join(paragraphs: list[str]) -> str:
    return "\n".join(paragraphs)


def fit_tfidf_svm(documents: list[list[str]], targets: np.ndarray, single_label: bool, seed: int):
    """Word and bigram TF–IDF with a linear SVM."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    features = vectorizer.fit_transform([_join(doc) for doc in documents])
    if single_label:
        classifier = LinearSVC(random_state=seed)
    else:
        classifier = OneVsRestClassifier(LinearSVC(random_state=seed))
    classifier.fit(features, targets)
    return vectorizer, classifier


def predict_tfidf_svm(vectorizer, classifier, documents: list[list[str]], single_label: bool) -> np.ndarray:
    features = vectorizer.transform([_join(doc) for doc in documents])
    if single_label:
        return classifier.predict(features)
    return classifier.predict(features).astype(np.int64)
