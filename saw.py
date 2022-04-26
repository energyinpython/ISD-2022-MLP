import numpy as np

from normalizations import minmax_normalization
from mcdm_method import MCDM_method


class SAW(MCDM_method):
    def __init__(self, normalization_method = minmax_normalization):
        self.normalization_method = normalization_method

    def __call__(self, matrix, weights, types):
        SAW._verify_input_data(matrix, weights, types)
        return SAW._saw(matrix, weights, types, self.normalization_method)

    @staticmethod
    def _saw(matrix, weights, types, normalization_method):
        # Normalize matrix using chosen normalization (for example linear normalization)
        norm_matrix = normalization_method(matrix, types)

        # Multiply all rows of normalized matrix by weights
        weighted_matrix = norm_matrix * weights

        # aggregate and return scores
        return np.sum(weighted_matrix, axis = 1)