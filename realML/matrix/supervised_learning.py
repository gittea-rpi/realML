import abc

from . import base

__all__ = ('SupervisedLearningPrimitiveBase',)


class SupervisedLearningPrimitiveBase(base.PrimitiveBase):
    def __init__(self):
        """
        Initializes the supervised learning primitive.

        All primitives should specify all the hyper-parameters that can be set at the class
        level in their ``__init__`` as explicit keyword arguments (no ``*args`` or ``**kwargs``).
        Available hyper-parameters should be specified in primitive's D3M annotation.
        """

    @abc.abstractmethod
    def fit(self, X, y, sample_weight=None, classes=None):
        """
        Build a model from the training set (X, y).

        Values in X and y do not have to be primitive values, but X and y can be multi-dimensional arrays.

        ``fit`` can be called multiple times and model is further trained with every call.

        Argument ``classes`` can be optionally set to inform the model about classes not available in
        the first call to ``fit``.

        Subclasses can accept custom additional but explicitly defined keyword arguments,
        which should be specified in primitive's D3M annotation.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features, ...]
            The training input samples.
        y : array-like, shape = [n_samples, ...] or [n_samples, n_outputs, ...]
            The target values (class labels).
        sample_weight : array-like, shape = [n_samples], optional
            Sample weights. If None, then samples are equally weighted.
        classes : array of shape = [n_classes, ...] or a list of such arrays, optional
            The classes labels (single output problem), or a list of arrays of class labels (multi-output problem).
            This informs the model about classes not available in the first call to ``fit``.
        """

    @abc.abstractmethod
    def predict(self, X):
        """
        Predict value for each sample in X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features, ...]
            The input samples.

        Returns
        -------
        array of shape = [n_samples, ...] or [n_samples, n_outputs, ...]
            The predictions.
        """

    @abc.abstractmethod
    def predict_log_proba(self, X):
        """
        Predict class log-probabilities of the input samples X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features, ...]
            The input samples.

        Returns
        -------
        array of shape = [n_samples, n_classes], or a list of n_outputs such arrays if n_outputs > 1
            The class probabilities of the input samples. The order of the classes
            corresponds to that in the argument ``classes`` to ``fit`` method.
        """

    def staged_fit(self, X, y, sample_weight=None, classes=None, **kwargs):
        """
        An iterative version of ``fit`` which yields at each internal iteration, for example,
        after every batch of training. This allows incremental monitoring and evaluation of
        the training process. When a fit generator yields, it should be possible to call
        methods on the instance and operate on current state of the model (at the time of
        yielding), including pickling the instance.
        """

        yield self.fit(X, y, sample_weight, classes, **kwargs)

    def staged_predict(self, X):
        """
        An iterative version of ``predict`` which yields at each internal iteration, for example,
        if predictions is gradually improving. At every iteration, returned value should be of
        the same shape as the return value for ``predict``, but some elements might be missing.
        """

        yield self.predict(X)

    def staged_predict_log_proba(self, X):
        """
        An iterative version of ``predict_log_proba`` which yields at each internal iteration, for
        example, if predictions is gradually improving. At every iteration, returned value should be of
        the same shape as the return value for ``predict_log_proba``, but some elements might be missing.
        """

        yield self.predict_log_proba(X)

