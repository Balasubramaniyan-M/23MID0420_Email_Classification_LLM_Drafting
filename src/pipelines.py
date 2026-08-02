"""
TF-IDF pipeline registry for sparse text classification.
Enforces that feature extraction fitting occurs strictly inside pipeline transformers during cross-validation.
"""

from typing import Dict, Any
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

RANDOM_STATE = 42


def make_pipeline(classifier: Any) -> Pipeline:
    """
    Construct a scikit-learn Pipeline combining common TF-IDF vectorization with a classifier.

    Parameters
    ----------
    classifier : Any
        Scikit-learn compatible estimator object.

    Returns
    -------
    Pipeline
        Unfitted scikit-learn Pipeline instance.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=60000,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def get_model_registry(random_state: int = RANDOM_STATE) -> Dict[str, Pipeline]:
    """
    Get dictionary of all 5 benchmark classifier pipelines required by Lab 03.

    Parameters
    ----------
    random_state : int, optional
        Seed for reproducibility, defaults to 42.

    Returns
    -------
    Dict[str, Pipeline]
        Dictionary mapping model short-names to unfitted Pipeline objects.
    """
    return {
        "dummy_majority": make_pipeline(
            DummyClassifier(strategy="most_frequent")
        ),
        "multinomial_nb": make_pipeline(
            MultinomialNB(alpha=1.0)
        ),
        "complement_nb": make_pipeline(
            ComplementNB(alpha=1.0)
        ),
        "logistic_regression": make_pipeline(
            LogisticRegression(
                max_iter=2500, class_weight="balanced", random_state=random_state
            )
        ),
        "linear_svc": make_pipeline(
            LinearSVC(
                class_weight="balanced", random_state=random_state
            )
        ),
    }
