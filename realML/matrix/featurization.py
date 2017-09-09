import abc

from . import base

__all__ = ('FeaturizationPrimitiveBase',)


class FeaturizationPrimitiveBase(base.PrimitiveBase):

    def __init__(self):
        """
        Initializes the featurization primitive.

        All primitives should specify all the hyper-parameters that can be set at the class
        level in their ``__init__`` as explicit keyword arguments (no ``*args`` or ``**kwargs``).
        Available hyper-parameters should be specified in primitive's D3M annotation.
        """

    @abc.abstractmethod
    def fit(self, intype, data, labels=None):
        """
        Takes input data in the specified format and performs the featurization computation, possibly including model-fitting
        (e.g. for deep-learning based featurization primitives).

        Parameters
        ----------

            intype: specifies the format of the input data

                Possible intype formats are:
                "image": a list of 1 or more images, each an m x n x 3 unsigned integer array-like (m and n need not necessarily be the same for all images)

                "video": a list of 1 or more videos, each an m x n x t x 3 unsigned integer array-like (m, n and t need not necessarily be the same for all videos)

                "dict": a catch-all type for idiosyncratic data formats, such as tabular data.  The exact dictionary fields required will be specific to
                individual featurization primitives

                "time_series": a list of 1 or more numerical time-series, each a 1-dimensional array-like (this type covers and includes audio data)

                "time_series_cat": a list of 1 or more categorical time series, each an unsigned integer array-like

                "text": a list of 1 or more documents, with each document object consisting of a string containing the document's text

                "matrix": a precomputed affinity or co-occurrence matrix, represented as a 2-d float array-like

                Each primitive should specify applicable input data formats in its documentation and primitive annotation (annotation fields need to be worked out
                with JPL)

            data: the input data in the specified input format

            labels: optional argument, used in cases where the featurization primitive will be doing supervised model-fitting.
                Should be an array-like containing categorical labels or regression targets for each input data instance.
        """

    @abc.abstractmethod
    def predict(self, outtype, data=None):
        """
        Returns (and possibly computes, if the fit method actually involved model-fitting) the computed features in a specified output format.

        Parameters
        ----------
            outtype: specifies the format of the output data

                Possible outtype formats are:
                "array1": a 1-dimensional array-like (probably for single-instance featurization only)

                "array2": a 2-dimensional array-like, with instances in rows and features in columns

                "array2+S": 2 outputs - [0] is a 2-dimensional array-like, with instances in rows and features in columns
                     and [1] is the ordered list of instances that were successfully featurized.  Instances not listed in [1]
                     (due to insufficient data, etc.) will be excluded from [0]

                "array2+N": arbitrary number of outputs - [0] is the number of returned array-likes, [1]-[n] are 2-dimensional arrays-likes.
                    The exact meaning of these arrays will need to be primitive-specific, but this could be used to return multiple embeddings/feature sets

                "array2+names": 2 outputs - [0] is a list of feature names/descriptions and [1] is a 2D array-like with instances in rows and features in columns

                Each primitive should specify applicable output data formats in its documentation and primitive annotation (annotation fields need to
                be worked out with JPL)

            data: optional argument, used in cases where the original input data was used to train a featurization model, and featurization will now be
                performed on a new set of test data

        Returns
        -------
            featurized_data: the computed feature set, in the specified output format
        """
