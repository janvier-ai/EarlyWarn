# EarlyWarner
# Alert before potential issues arise in a system.
# Is either seperable or end-to-end.
# If seperable, it has a feature extractor and/or only a collection of classifiers.
# If end-to-end, it has a single model.
# Takes a collection of feature extractors, classifiers, and a trigger
# a mode either fixed deadline or moving window
# Has methods to fit, predict and evaluate.
# Can save and load models.
# Can handle data in batches.
# Can handle multi-class classification.
# Can handle multi-label classification.
# _extractor_fitted: boolean, indicates if the feature extractor and/or classifier have been pre-fitted.
# _classifier_fitted: boolean, indicates if the classifier has been pre-fitted.
# _fitted: boolean, indicates if the EarlyWarner has been fitted.

class EarlyWarner:
    pass