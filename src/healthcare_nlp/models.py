from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


@dataclass
class ModelSpec:
    name: str
    pipeline: Pipeline
    param_grid: dict


def build_model_specs(random_state: int = 42) -> list[ModelSpec]:
    common_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        max_features=40000,
        sublinear_tf=True,
    )

    specs = [
        ModelSpec(
            name="logistic_regression",
            pipeline=Pipeline(
                [
                    ("tfidf", common_vectorizer),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            solver="liblinear",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            param_grid={
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "clf__C": [0.5, 1.0, 2.0],
            },
        ),
        ModelSpec(
            name="linear_svm",
            pipeline=Pipeline(
                [
                    ("tfidf", common_vectorizer),
                    ("clf", LinearSVC(class_weight="balanced", random_state=random_state)),
                ]
            ),
            param_grid={
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "clf__C": [0.5, 1.0, 2.0],
            },
        ),
        ModelSpec(
            name="random_forest",
            pipeline=Pipeline(
                [
                    ("tfidf", common_vectorizer),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=250,
                            class_weight="balanced_subsample",
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            param_grid={
                "clf__max_depth": [None, 50],
                "clf__min_samples_split": [2, 5],
            },
        ),
        ModelSpec(
            name="naive_bayes",
            pipeline=Pipeline(
                [
                    ("tfidf", common_vectorizer),
                    ("clf", MultinomialNB()),
                ]
            ),
            param_grid={
                "clf__alpha": [0.3, 1.0, 2.0],
            },
        ),
    ]
    return specs


def tune_model(spec: ModelSpec, x_train, y_train, cv_folds: int = 3):
    search = GridSearchCV(
        estimator=spec.pipeline,
        param_grid=spec.param_grid,
        scoring="f1",
        cv=cv_folds,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(x_train, y_train)
    return search
