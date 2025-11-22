"""Progressive Prefix Ensemble Classifier.

Classifier that implements a progressive prefix ensemble strategy
based on a given base classifier.
"""

__maintainer__ = []
__all__ = ["ProgressivePrefixEnsembleClassifier"]

import copy

import numpy as np
from joblib import Parallel, delayed
from sklearn.utils import check_random_state

from aeon.base._base import _clone_estimator
from aeon.classification.base import BaseClassifier
from aeon.classification.convolution_based import MiniRocketClassifier
from aeon.utils.validation import check_n_jobs

class ProgressivePrefixEnsembleClassifier(BaseClassifier):
    """
    Progressive Prefix Ensemble Classifier.

    Classifier that implements a progressive prefix ensemble strategy 
    based on a given base classifier.

    Overview: 
        Build n classifiers, where n is the number of prefixes.

    Parameters
    ----------
    estimator : aeon classifier, default=None
        An aeon estimator to be built using the transformed data. Defaults to a
        default MiniRocketClassifier classifier.
    classification_points : List or None, default=None
        List of integer time series time stamps to build classifiers and allow
        predictions at. Early predictions must have a series length that matches a value
        in the _classification_points List. Duplicate values will be removed, and the
        full series length will be appended if not present.
        If None, will use 10 thresholds equally spaces from 0 to the series length.
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

    See Also
    --------

    Notes
    -----

    References
    ----------

    Examples
    --------
    >>> from earlywarn.classification.compose import ProgressivePrefixEnsembleClassifier
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12,
    ...                              return_y=True, random_state=0)
    >>> clf = ProgressivePrefixEnsembleClassifier(random_state=0)
    >>> clf.fit(X, y)
    ProgressivePrefixEnsembleClassifier(random_state=0)
    >>> clf.predict(X)
    array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0])
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "capability:missing_values": True,
        "capability:unequal_length": True,
    }

    def __init__(
        self,
        estimator=None,
        classification_points=None,
        n_jobs=1,
        random_state=None,
    ):
        self.estimator = estimator
        self.classification_points = classification_points

        self.n_jobs = n_jobs
        self.random_state = random_state

        self._estimators = []
        self._classification_points = []

        self.n_cases_ = 0
        self.n_channels_ = 0
        self.n_timepoints_ = 0

        super().__init__()

    def _fit(self, X, y):
        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape
        self._n_jobs = check_n_jobs(self.n_jobs)

        self._estimator = (
            MiniRocketClassifier() if self.estimator is None else self.estimator
        )

        m = getattr(self._estimator, "predict_proba", None)
        if not callable(m):
            raise ValueError("Base estimator must have a predict_proba method.")

        self._classification_points = (
            copy.deepcopy(self.classification_points)
            if self.classification_points is not None
            else np.arange(stop=self.n_timepoints_, step=10).tolist()
        )
        # remove duplicates
        self._classification_points = list(set(self._classification_points))
        self._classification_points.sort()
        # remove classification points that are less than 3 time stamps
        self._classification_points = [i for i in self._classification_points if i >= 3]
        # make sure the full series length is included
        if self._classification_points[-1] != self.n_timepoints_:
            self._classification_points.append(self.n_timepoints_)
        # create dictionary of classification point indices
        self._classification_point_dictionary = {}
        for index, classification_point in enumerate(self._classification_points):
            self._classification_point_dictionary[classification_point] = index

        # avoid nested parallelism
        m = getattr(self._estimator, "n_jobs", None)
        threads = self._n_jobs if m is None else 1

        rng = check_random_state(self.random_state)

        self._estimators = Parallel(n_jobs=threads, prefer="threads")(
            delayed(self._fit_estimator)(
                X,
                y,
                i,
                check_random_state(rng.randint(np.iinfo(np.int32).max)),
            )
            for i in range(len(self._classification_points))
        )

        return self

    def _fit_estimator(self, X, y, i, rng):
        estimator = _clone_estimator(
            self._estimator,
            rng,
        )

        m = getattr(estimator, "n_jobs", None)
        if m is not None:
            estimator.n_jobs = self._n_jobs

        # filter np.nan from X for this prefix
        X = X[:, :, : self._classification_points[i]]
        mask = ~np.isnan(X).any(axis=(1, 2))
        X = X[mask]
        y = y[mask]

        # fit estimator for this threshold
        estimator.fit(X[:, :, : self._classification_points[i]], y)

        return estimator
    
    def _predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        probas = self._predict_proba(X)
        return self._proba_output_to_preds(probas)
    
    def _predict_proba(self, X) -> tuple[np.ndarray, np.ndarray]:
        n_cases, _, n_timepoints = X.shape

        # maybe use the largest index that is smaller than the series length
        next_idx = self._get_next_idx(n_timepoints) + 1

        # if the input series length is invalid
        if next_idx == 0:
            raise ValueError(
                f"Input series length does not match the classification points produced"
                f" in fit. Input series length must be greater then the first point. "
                f"Current classification points: {self._classification_points}"
            )

        # avoid nested parallelism
        m = getattr(self._estimator, "n_jobs", None)
        threads = self._n_jobs if m is None else 1

        rng = check_random_state(self.random_state)

        # compute all new updates since then
        probas = Parallel(n_jobs=threads, prefer="threads")(
            delayed(self._predict_proba_for_estimator)(
                X,
                i,
                check_random_state(rng.randint(np.iinfo(np.int32).max)),
            )
            for i in range(0, next_idx)
        )

        return probas
    
    def _proba_output_to_preds(self, out):
        rng = check_random_state(self.random_state)
        preds = [np.array(
            [
                (
                    self.classes_[
                        int(rng.choice(np.flatnonzero(prob == prob.max())))
                    ]
                )
                for prob in o
            ]
        ) for o in out]
        return preds, out
    
    def _predict_proba_for_estimator(self, X, i, rng):
        probas = self._estimators[i].predict_proba(
            X[:, :, : self._classification_points[i]]
        )

        return probas
    
    def _get_next_idx(self, n_timepoints):
        """Return the largest index smaller than the series length."""
        next_idx = -1
        for idx, offset in enumerate(np.sort(self._classification_points)):
            if offset <= n_timepoints:
                next_idx = idx
        return next_idx