import numpy as np


# linear normalization
def linear_normalization(matrix, types):
    """
    Normalize decision matrix using linear normalization method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns
        types : ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

    Returns
    --------
        ndarray
            Normalized decision matrix

    Examples
    ----------
    >>> nmatrix = linear_normalization(matrix, types)
    """

    ind_profit = np.where(types == 1)[0]
    ind_cost = np.where(types == -1)[0]
    x_norm = np.zeros(np.shape(matrix))

    x_norm[:, ind_profit] = matrix[:, ind_profit] / (np.amax(matrix[:, ind_profit], axis = 0))
    x_norm[:, ind_cost] = np.amin(matrix[:, ind_cost], axis = 0) / matrix[:, ind_cost]
    return x_norm


# min-max normalization
def minmax_normalization(matrix, types):
    """
    Normalize decision matrix using minimum-maximum normalization method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns
        types : ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

    Returns
    --------
        ndarray
            Normalized decision matrix

    Examples
    ----------
    >>> nmatrix = minmax_normalization(matrix, types)
    """

    x_norm = np.zeros((matrix.shape[0], matrix.shape[1]))
    ind_profit = np.where(types == 1)[0]
    ind_cost = np.where(types == -1)[0]

    x_norm[:, ind_profit] = (matrix[:, ind_profit] - np.amin(matrix[:, ind_profit], axis = 0)
                             ) / (np.amax(matrix[:, ind_profit], axis = 0) - np.amin(matrix[:, ind_profit], axis = 0))

    x_norm[:, ind_cost] = (np.amax(matrix[:, ind_cost], axis = 0) - matrix[:, ind_cost]
                           ) / (np.amax(matrix[:, ind_cost], axis = 0) - np.amin(matrix[:, ind_cost], axis = 0))

    return x_norm


# max normalization
def max_normalization(matrix, types):
    """
    Normalize decision matrix using maximum normalization method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns
        types : ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

    Returns
    --------
        ndarray
            Normalized decision matrix

    Examples
    ----------
    >>> nmatrix = max_normalization(matrix, types)
    """

    maximes = np.amax(matrix, axis=0)
    ind = np.where(types == -1)[0]
    matrix = matrix/maximes
    matrix[:,ind] = 1-matrix[:,ind]
    return matrix


# sum normalization
def sum_normalization(matrix, types):
    """
    Normalize decision matrix using sum normalization method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns
        types : ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

    Returns
    --------
        ndarray
            Normalized decision matrix

    Examples
    ----------
    >>> nmatrix = sum_normalization(matrix, types)
    """

    x_norm = np.zeros((matrix.shape[0], matrix.shape[1]))
    ind_profit = np.where(types == 1)[0]
    ind_cost = np.where(types == -1)[0]

    x_norm[:, ind_profit] = matrix[:, ind_profit] / np.sum(matrix[:, ind_profit], axis = 0)

    x_norm[:, ind_cost] = (1 / matrix[:, ind_cost]) / np.sum((1 / matrix[:, ind_cost]), axis = 0)

    return x_norm


# vector normalization
def vector_normalization(matrix, types):
    """
    Normalize decision matrix using vector normalization method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns
        types : ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

    Returns
    --------
        ndarray
            Normalized decision matrix

    Examples
    -----------
    >>> nmatrix = vector_normalization(matrix, types)
    """

    x_norm = np.zeros((matrix.shape[0], matrix.shape[1]))
    ind_profit = np.where(types == 1)[0]
    ind_cost = np.where(types == -1)[0]

    x_norm[:, ind_profit] = matrix[:, ind_profit] / (np.sum(matrix[:, ind_profit] ** 2, axis = 0))**(0.5)

    x_norm[:, ind_cost] = 1 - (matrix[:, ind_cost] / (np.sum(matrix[:, ind_cost] ** 2, axis = 0))**(0.5))

    return x_norm
