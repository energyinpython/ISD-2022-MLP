import numpy as np

# reverse = True: descending order (TOPSIS, CODAS, SAW), False: ascending order (VIKOR, SPOTIS)
def rank_preferences(pref, reverse = True):
    """
    Rank alternatives according to MCDA preference function values. If more than one alternative
    have the same preference function value, they will be given the same rank value (tie).

    Parameters
    ------------
        pref : ndarray
            Vector with MCDA preference function values for alternatives
        reverse : bool
            The boolean variable is True for MCDA methods that rank alternatives in descending
            order (for example, TOPSIS, CODAS, SAW) and False for MCDA methods that rank alternatives in ascending
            order (for example, VIKOR, SPOTIS)
    
    Returns
    ---------
        ndarray
            Vector with alternatives ranking. Alternative with 1 value is the best and has the first position in the ranking.
    
    Examples
    ----------
    >>> rank = rank_preferences(pref, reverse = True)
    """

    rank = np.zeros(len(pref))
    sorted_pref = sorted(pref, reverse = reverse)
    pos = 1
    for i in range(len(sorted_pref) - 1):
        ind = np.where(sorted_pref[i] == pref)[0]
        rank[ind] = pos
        if sorted_pref[i] != sorted_pref[i + 1]:
            pos += 1
    ind = np.where(sorted_pref[i + 1] == pref)[0]
    rank[ind] = pos
    return rank.astype(int)
