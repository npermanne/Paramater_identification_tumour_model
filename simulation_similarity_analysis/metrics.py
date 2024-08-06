from skimage.metrics import structural_similarity
from enum import Enum
import math
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from scipy.spatial.distance import cosine


def ssim_function(image1, image2):
    min_value, max_value = min(np.min(image1), np.min(image2)), max(np.max(image1), np.max(image2))
    return structural_similarity(image1, image2, full=True, data_range=max_value - min_value)[0]


def corr_hist_function(image1, image2):
    min_value, max_value = min(np.min(image1), np.min(image2)), max(np.max(image1), np.max(image2))
    hist1, bins1 = np.histogram(image1, bins=np.linspace(min_value, max_value + 1, 100))
    hist2, bins2 = np.histogram(image2, bins=np.linspace(min_value, max_value + 1, 100))
    return np.corrcoef(hist1, hist2)[0, 1]


def inter_hist_function(image1, image2):
    min_value, max_value = min(np.min(image1), np.min(image2)), max(np.max(image1), np.max(image2))
    hist1, bins1 = np.histogram(image1, bins=np.linspace(min_value, max_value + 1, 100))
    hist2, bins2 = np.histogram(image2, bins=np.linspace(min_value, max_value + 1, 100))
    return np.sum(np.minimum(hist1, hist2)) / np.sum(hist1)


def set_function(image1, image2, func):
    i1, i2 = image1.flatten(), image2.flatten()
    c1, e1, h1 = i1 == -1, i1 == 0, i1 == 1
    c2, e2, h2 = i2 == -1, i2 == 0, i2 == 1

    c_index = func(c1, c2) if c1.any() or c2.any() else 0
    e_index = func(e1, e2) if e1.any() or e2.any() else 0
    h_index = func(h1, h2) if h1.any() or h2.any() else 0
    return (c_index + e_index + h_index) / 3


dice = lambda a, b: 2 * np.sum(a & b) / (np.sum(a) + np.sum(b))
jaccard = lambda a, b: np.sum(a & b) / (np.sum(a) + np.sum(b) - np.sum(a & b))


def corr(a,b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.corrcoef(a.flatten(), b.flatten())[0, 1]


class SimilarityMetric(Enum):
    IMAGE_ABSOLUTE_DIFFERENCE = 0
    CORRELATION_HISTOGRAM = 1
    SSIM = 2
    MEAN_ABSOLUTE_ERROR = 3
    ROOT_MEAN_SQUARED_ERROR = 4
    MAX_ABSOLUTE_ERROR = 5
    CORRELATION = 6
    MUTUAL_INFORMATION = 8
    COSINE_SIMILARITY = 9
    INTERSECTION_HISTOGRAM = 10
    DICE = 11
    JACCARD = 12

    def __str__(self):
        if self == SimilarityMetric.IMAGE_ABSOLUTE_DIFFERENCE:
            return "image absolute difference"
        elif self == SimilarityMetric.CORRELATION_HISTOGRAM:
            return "histogram correlation"
        elif self == SimilarityMetric.SSIM:
            return "ssim index"
        elif self == SimilarityMetric.MEAN_ABSOLUTE_ERROR:
            return "mean absolute error"
        elif self == SimilarityMetric.ROOT_MEAN_SQUARED_ERROR:
            return "root mean squared error"
        elif self == SimilarityMetric.MAX_ABSOLUTE_ERROR:
            return "max absolute error"
        elif self == SimilarityMetric.CORRELATION:
            return "correlation"
        elif self == SimilarityMetric.MUTUAL_INFORMATION:
            return "mutual information"
        elif self == SimilarityMetric.COSINE_SIMILARITY:
            return "cosine similarity"
        elif self == SimilarityMetric.INTERSECTION_HISTOGRAM:
            return "histogram intersection "
        elif self == SimilarityMetric.DICE:
            return "dice-Sørensen coefficient"
        elif self == SimilarityMetric.JACCARD:
            return "jaccard coefficient"

    def get_function(self):
        if self == SimilarityMetric.IMAGE_ABSOLUTE_DIFFERENCE:
            return lambda a, b: np.absolute(a - b)
        elif self == SimilarityMetric.CORRELATION_HISTOGRAM:
            return corr_hist_function
        elif self == SimilarityMetric.SSIM:
            return ssim_function
        elif self == SimilarityMetric.MEAN_ABSOLUTE_ERROR:
            return lambda a, b: -np.abs(a - b).mean()
        elif self == SimilarityMetric.ROOT_MEAN_SQUARED_ERROR:
            return lambda a, b: -math.sqrt(np.square(a - b).mean())
        elif self == SimilarityMetric.MAX_ABSOLUTE_ERROR:
            return lambda a, b: -np.max(np.absolute(a - b))
        elif self == SimilarityMetric.CORRELATION:
            return corr
        elif self == SimilarityMetric.MUTUAL_INFORMATION:
            return lambda a, b: mutual_info_regression(np.array([a.flatten()]).transpose(), b.flatten(), discrete_features='auto')[0]
        elif self == SimilarityMetric.COSINE_SIMILARITY:
            return lambda a, b: 1 - cosine(a.flatten(), b.flatten())
        elif self == SimilarityMetric.INTERSECTION_HISTOGRAM:
            return inter_hist_function
        elif self == SimilarityMetric.DICE:
            return lambda a, b: set_function(a, b, func=dice)
        elif self == SimilarityMetric.JACCARD:
            return lambda a, b: set_function(a, b, func=jaccard)


if __name__ == '__main__':
    a = np.array([
        [-1, -1],
        [-1, 0]
    ])
    b = np.array([
        [1, -1],
        [-1, 0]
    ])

    print(SimilarityMetric.DICE.get_function()(a, b))
