"""Timestamp Early Classifier.

An early classifier that makes a decision at a specified timestamp.
"""

__maintainer__ = []
__all__ = ["ALAP"]

from earlywarn.classification.early_classification._base import BaseEarlyClassifier

class TimestampEarlyClassifier(BaseEarlyClassifier):
    """
    Timestamp Early Classifier.

    An early classifier that makes a decision at a specified timestamp.

    Overview:
        Build n classifiers, where n is the number of classification_points.
        While a prediction is still deemed unsafe:
            Make a prediction using the series length at classification point i.
            Decide the prediction is safe if it corresponds to the specified timestamp.

    Parameters
    ----------
    timestamp_idx : int, default=-1
        The timestamp index at which to make a prediction. If -1, will use the full series length.
    estimator : aeon classifier, default=None
        An aeon estimator to be built using the transformed data. Defaults to a
        default DrCIF classifier.
    classification_points : List or None, default=None
        List of integer time series time stamps to build classifiers and allow
        predictions at. Early predictions must have a series length that matches a value
        in the _classification_points List. Duplicate values will be removed, and the
        full series length will be appended if not present.
        If None, will use 20 thresholds linearly spaces from 0 to the series length.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    n_cases_ : int
        The number of train cases.
    n_channels_ : int
        The number of dimensions per case.
    n_timepoints_ : int
        The full length of each series.
    classes_ : list
        The unique class labels.
    state_info : 2d np.ndarray (4 columns)
        Information stored about input instances after the decision-making process in
        update/predict methods. Used in update methods to make decisions based on
        the results of previous method calls.
        Records in order: the time stamp index, the number of consecutive decisions
        made, the predicted class and the series length.

    Examples
    --------
    >>> from aeon.classification.early_classification import (
    ...     TimestampEarlyClassifier,
    ... )
    >>> from aeon.classification.interval_based import TimeSeriesForestClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = TimestampEarlyClassifier(
    ...     classification_points=[6, 16, 24],
    ...     estimator=TimeSeriesForestClassifier(n_estimators=5),
    ... )
    >>> clf.fit(X_train, y_train)
    TimestampEarlyClassifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
    }

    def __init__(
        self,
        estimator=None,
        timestamp_idx=-1,
        classification_points=None,
        n_jobs=1,
        random_state=None,
    ):
        self.estimator = estimator
        self.timestamp_idx = timestamp_idx
        self.classification_points = classification_points

        self.n_jobs = n_jobs
        self.random_state = random_state

        self._estimators = []
        self._classification_points = []

        self.n_cases_ = 0
        self.n_channels_ = 0
        self.n_timepoints_ = 0

        super().__init__()

    