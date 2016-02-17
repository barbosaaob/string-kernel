import numpy as np


def substrings(s, p=2):
    """
    Computes substrings (contiguous) of size p.
    Input:
        s is a string or a list of strings.
        p is the size of the substrings.
    Returns:
        A list with all substrings of strings (or strings in list s).

    doctest:
    >>> substrings("kernel")
    ['el', 'er', 'ke', 'ne', 'rn']
    >>> substrings("kernel", p=3)
    ['ern', 'ker', 'nel', 'rne']
    >>> substrings("kernel", p=5)
    ['ernel', 'kerne']
    >>> substrings(["bar", "bat", "car", "cat"])
    ['ar', 'at', 'ba', 'ca']
    >>> substrings(["kernel", "implicit"], p=3)
    ['cit', 'ern', 'ici', 'imp', 'ker', 'lic', 'mpl', 'nel', 'pli', 'rne']
    >>> substrings("kernel", p=0)
    ['']
    >>> substrings(1)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (substrings): invalid input parameters.
    >>> substrings("implicit", 1.0)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (substrings): invalid input parameters.
    >>> substrings("implicit", -1)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (substrings): invalid input parameters.
    >>> substrings("kernel", p=7)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (substrings): len(s) < p.
    >>> substrings(["kernel", "implicit"], p=7)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (substrings): len(ss) < p.
    >>> substrings(["implicit", "kernel"], p=7)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (substrings): len(ss) < p.
    >>> substrings(["implicit", "kernel", "cat"], p=4)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (substrings): len(ss) < p.
    """
    assert ((type(s) is str) or (type(s) is list)) and (type(p) is int) and \
        (p >= 0), "*** ERROR (substrings): invalid input parameters."
    if type(s) is list:
        substrings = list()
        for ss in s:
            assert len(ss) >= p, "*** ERROR (substrings): len(ss) < p."
            nsubstrings = len(ss) - p + 1  # numumber of (contiguous) substr
            for i in range(nsubstrings):
                substrings.append(ss[i:i+p])
        substrings = list(set(substrings))
        substrings.sort()
    elif type(s) is str:
        assert len(s) >= p, "*** ERROR (substrings): len(s) < p."
        nsubstrings = len(s) - p + 1  # number of (contiguous) substrings
        substrings = list()
        for i in range(nsubstrings):
            substrings.append(s[i:i+p])
        substrings = list(set(substrings))
        substrings.sort()

    return substrings


def p_spectrum(s, t, substr):
    """
    Please use p_spectrum_r(s, t, p) function instead.

    Computes p-spectrum kernel.
    Input:
        s is a string.
        t is a string.
        substr is a list with all substrings of a known dictionary.
    Returns:
        The value of the kernel between s and t.

    doctest:
    >>> substr = substrings(["car", "cat", "bar", "bat"], p=2)
    >>> p_spectrum("cat", "car", substr)
    1.0
    >>> p_spectrum("cat", "bat", substr)
    1.0
    >>> p_spectrum("cat", "bar", substr)
    0.0
    >>> p_spectrum("cat", "cat", substr)
    2.0
    >>> substr = substrings(["car", "cat", "bar", "bat"], p=3)
    >>> p_spectrum("cat", "cat", substr)
    1.0
    >>> p_spectrum("cat", "car", substr)
    0.0
    >>> substr = substrings(["statistics", "computation"], p=3)
    >>> p_spectrum("statistics", "computation", substr)
    2.0
    >>> p_spectrum(1, "a", substr)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_spectrum): invalid input parameters.
    >>> p_spectrum("1", 2, substr)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_spectrum): invalid input parameters.
    >>> p_spectrum("1", "b", "a")
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_spectrum): invalid input parameters.
    """
    assert (type(s) is str) and (type(t) is str) and (type(substr) is list), \
        "*** ERROR (p_spectrum): invalid input parameters."
    from re import finditer
    phi_s = np.zeros(len(substr),)
    phi_t = np.zeros(len(substr),)
    for i in range(len(substr)):
        s_occurr = [o for o in finditer(substr[i], s)]
        t_occurr = [o for o in finditer(substr[i], t)]
        phi_s[i] = len(s_occurr)
        phi_t[i] = len(t_occurr)

    return phi_s.dot(phi_t)


def p_suffix_kernel(s, t, p=2):
    """
    Computes p-suffix kernel.
    Input:
        s is a string.
        t is a string.
        p is the size of the substrings.
    Returns:
        The valure of the p-suffix kernel betwwen s and t.

    doctest:
    >>> p_suffix_kernel("kernel", "tonel", p=3)
    1
    >>> p_suffix_kernel("kernel", "tonel", p=2)
    1
    >>> p_suffix_kernel("kernel", "tonel", p=1)
    1
    >>> p_suffix_kernel("kernel", "cat")
    0
    >>> p_suffix_kernel("cat", "cat", p=0)
    0
    >>> p_suffix_kernel(1, "cat")
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_suffix_kernel): invalid input parameters.
    >>> p_suffix_kernel("cat", ["cat"])
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_suffix_kernel): invalid input parameters.
    >>> p_suffix_kernel("cat", "cat", 2.0)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_suffix_kernel): invalid input parameters.
    >>> p_suffix_kernel("cat", "cat", -1)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_suffix_kernel): invalid input parameters.
    >>> p_suffix_kernel("cat", "abcd", 4)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_suffix_kernel): len(s) or len(t) < p.
    >>> p_suffix_kernel("abcd", "cat", 4)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_suffix_kernel): len(s) or len(t) < p.
    """
    assert (type(s) is str) and (type(t) is str) and (type(p) is int) and \
        (p >= 0), "*** ERROR (p_suffix_kernel): invalid input parameters."
    assert (len(s) >= p) and (len(t) >= p), \
        "*** ERROR (p_suffix_kernel): len(s) or len(t) < p."
    if (p == 0):
        return 0
    s_suffix = s[-p:]
    t_suffix = t[-p:]

    return int(s_suffix == t_suffix)


def p_spectrum_r(s, t, p=2):
    """
    Computes the p-spectrum kernel recursivelly.
    Input:
        s is a string.
        t is a string.
        p is the size of the substrings.
    Returns:
        The valure of the p-spectrum kernel between s and t.

    doctest:
    >>> p_spectrum_r("statistics", "computation", p=3)
    2.0
    >>> p_spectrum_r("cat", "car")
    1.0
    >>> p_spectrum_r("cat", "bat")
    1.0
    >>> p_spectrum_r("cat", "bar")
    0.0
    >>> p_spectrum_r("cat", "cat")
    2.0
    >>> p_spectrum_r("cat", "cat", p=0)
    0.0
    >>> p_spectrum_r("cat", "cat", p=3)
    1.0
    >>> p_spectrum_r("cat", "car", p=3)
    0.0
    >>> p_spectrum_r(1, "cat")
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_spectrum_r): invalid input parameters.
    >>> p_spectrum_r("cat", 1)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_spectrum_r): invalid input parameters.
    >>> p_spectrum_r("cat", "cat", 2.0)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_spectrum_r): invalid input parameters.
    >>> p_spectrum_r("cat", "cat", -2)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_spectrum_r): invalid input parameters.
    >>> p_spectrum_r("cat", "abcd", 4)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_spectrum_r): len(s) or len(t) < p.
    >>> p_spectrum_r("abcd", "cat", 4)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_spectrum_r): len(s) or len(t) < p.
    """
    assert (type(s) is str) and (type(t) is str) and (type(p) is int) and \
        (p >= 0), "*** ERROR (p_spectrum_r): invalid input parameters."
    assert (len(s) >= p) and (len(t) >= p), \
        "*** ERROR (p_spectrum_r): len(s) or len(t) < p."
    s_range = range(len(s) - p + 1)
    t_range = range(len(t) - p + 1)
    sum = 0.0
    for i in s_range:
        for j in t_range:
            sum += p_suffix_kernel(s[i:i+p], t[j:j+p], p)

    return sum


def p_suffix_blended(s, t, lamb=1, p=2):
    """
    Computes "blended" p-suffix kernel, refer to Remark 11.13.

    No tests availiable yet (hope it is bug free :P).
    It does not crash at least.
    >>> p_suffix_blended("cat", "cat", lamb=1, p=2)
    2
    >>> p_suffix_blended("statistics", "computation")
    0

    doctest:
    >>> p_suffix_blended(1, "cat")
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_suffix_blended): invalid input parameters.
    >>> p_suffix_blended("cat", 1)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_suffix_blended): invalid input parameters.
    >>> p_suffix_blended("cat", "car", p=-1)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_suffix_blended): invalid input parameters.
    >>> p_suffix_blended("cat", "car", p=1.0)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_suffix_blended): invalid input parameters.
    >>> p_suffix_blended("cat", "abcd", p=4)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_suffix_blended): len(s) or len(t) < p.
    >>> p_suffix_blended("abcd", "cat", p=4)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_suffix_blended): len(s) or len(t) < p.
    """
    assert (type(s) is str) and (type(t) is str) and (type(p) is int) \
        and (p >= 0), "*** ERROR (p_suffix_blended): invalid input parameters."
    assert (len(s) >= p) and (len(t) >= p), \
        "*** ERROR (p_suffix_blended): len(s) or len(t) < p."
    if (p == 0):
        return 0
    s_suffix = s[-p:]
    t_suffix = t[-p:]
    if (s_suffix == t_suffix):
        return lamb**p * (1 + p_suffix_blended(s, t, lamb, p-1))

    return 0


def p_spectrum_blended(s, t, lamb=1, p=2):
    """
    Computes "blended" p-spectrum kernel, refer to Remaerk 11.13.

    No conclusive tests availiable yet (hope it is bug free :P).
    >>> p_spectrum_blended("cat", "cat")
    4.0
    >>> p_spectrum_blended("cat", "tac")
    0.0
    >>> p_spectrum_blended("statistics", "computation")
    8.0
    >>> p_spectrum_blended("statistics", "statistic")
    24.0
    >>> p_spectrum_blended("statistics", "statistics", p=4)
    28.0
    >>> p_spectrum_blended("statistics", "scitsitats", p=4)
    0.0

    doctest:
    >>> p_spectrum_blended(1, "cat")
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_spectrum_blended): invalid input parameters.
    >>> p_spectrum_blended("cat", 1)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_spectrum_blended): invalid input parameters.
    >>> p_spectrum_blended("cat", "car", p=-1)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_spectrum_blended): invalid input parameters.
    >>> p_spectrum_blended("cat", "car", p=1.0)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_spectrum_blended): invalid input parameters.
    >>> p_spectrum_blended("cat", "abcd", p=4)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_spectrum_blended): len(s) or len(t) < p.
    >>> p_spectrum_blended("abcd", "cat", p=4)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (p_spectrum_blended): len(s) or len(t) < p.
    """
    assert (type(s) is str) and (type(t) is str) and (type(p) is int) and \
        (p >= 0), "*** ERROR (p_spectrum_blended): invalid input parameters."
    assert (len(s) >= p) and (len(t) >= p), \
        "*** ERROR (p_spectrum_blended): len(s) or len(t) < p."
    s_range = range(len(s) - p + 1)
    t_range = range(len(t) - p + 1)
    sum = 0.0
    for i in s_range:
        for j in t_range:
            sum += p_suffix_blended(s[i:i+p], t[j:j+p], lamb, p)

    return sum


def subsequences(s, p=2):
    """
    doctest:
    >>> subsequences("cat")
    ['at', 'ca', 'ct']
    >>> subsequences("kernel", 0)
    ['']
    >>> subsequences("kernel", p=3)
    ['eel', 'ene', 'enl', 'ere', 'erl', 'ern', 'kee', 'kel', 'ken', 'ker', \
'kne', 'knl', 'kre', 'krl', 'krn', 'nel', 'rel', 'rne', 'rnl']
    >>> subsequences("kernel", p=4)
    ['enel', 'erel', 'erne', 'ernl', 'keel', 'kene', 'kenl', 'kere', 'kerl', \
'kern', 'knel', 'krel', 'krne', 'krnl', 'rnel']
    >>> subsequences(["bar", "baa", "car", "cat"], p=1)
    ['a', 'b', 'c', 'r', 't']
    >>> subsequences(["bar", "baa", "car", "cat"], p=2)
    ['aa', 'ar', 'at', 'ba', 'br', 'ca', 'cr', 'ct']
    >>> subsequences(["bar", "baa", "car", "cat"], p=3)
    ['baa', 'bar', 'car', 'cat']

    #>>> len(subsequences("statistics", p=2))
    #25
    #>>> len(subsequences("statistics", p=3))
    #120

    >>> subsequences(1)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (subsequences): invalid input parameters.
    >>> subsequences("cat", -1)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (subsequences): invalid input parameters.
    >>> subsequences("cat", 1.0)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (subsequences): invalid input parameters.
    >>> subsequences("kernel", p=7)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (subsequences): len(s) < p.
    >>> subsequences(["kernel", "implicit"], p=7)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (subsequences): len(ss) < p.
    >>> subsequences(["implicit", "kernel"], p=7)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (subsequences): len(ss) < p.
    >>> subsequences(["implicit", "kernel", "cat"], p=4)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (subsequences): len(ss) < p.
    """
    assert ((type(s) is str) or (type(s) is list)) and (type(p) is int) and \
        (p >= 0), "*** ERROR (subsequences): invalid input parameters."
    from itertools import combinations
    if (type(s) is list):
        subseq = list()
        for ss in s:
            assert (len(ss) >= p), "*** ERROR (subsequences): len(ss) < p."
            for i in combinations(ss, p):
                sss = str()
                for j in i:
                    sss += j
                subseq.append(sss)
            subseq = list(set(subseq))
            subseq.sort()
    elif (type(s) is str):
        assert (len(s) >= p), "*** ERROR (subsequences): len(s) < p."
        subseq = list()
        for i in combinations(s, p):
            ss = str()
            for j in i:
                ss += j
            subseq.append(ss)
        subseq = list(set(subseq))
        subseq.sort()

    return subseq


def all_subsequences(s, t, debug=False):
    """
    Computes the all subsequences kernel, refer to Algorithm 11.20.
    Input:
        s is a string.
        t is a string.
    Returns:
        The value of all subsequences kernel between s and t.

    doctest:
    >>> all_subsequences("gatta", "cata")
    14.0
    >>> all_subsequences("gatta", "cata", debug=True)
    [[  1.   1.   1.   1.   1.   1.]
     [  1.   1.   1.   1.   1.   1.]
     [  1.   1.   2.   2.   2.   3.]
     [  1.   1.   2.   4.   6.   7.]
     [  1.   1.   3.   5.   7.  14.]]
    14.0
    >>> all_subsequences("gatt", "cata")
    7.0
    >>> all_subsequences("gatt", "c")
    1.0
    >>> all_subsequences("gatt", "cat")
    6.0
    >>> all_subsequences("", "")
    1.0
    >>> all_subsequences("a", "")
    1.0
    >>> all_subsequences("", "a")
    1.0
    >>> all_subsequences("gatta", "pcpaptpa")
    14.0
    >>> all_subsequences(1, "cat")
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (all_subsequences): invalid input parameters.
    >>> all_subsequences("cat", 1)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (all_subsequences): invalid input parameters.
    """
    assert (type(s) is str) and (type(t) is str), \
        "*** ERROR (all_subsequences): invalid input parameters."
    if ((s == "") or (t == "")):
        return 1.0
    n = len(s)
    m = len(t)
    DP = np.ones((n+1, m+1))
    P = np.zeros(m+1,)
    for i in range(1, n+1):
        last = 0
        P[0] = 0
        for k in range(1, m+1):
            P[k] = P[last]
            if t[k-1] == s[i-1]:
                P[k] = P[last] + DP[i-1, k-1]
                last = k
        for k in range(1, m+1):
            DP[i, k] = DP[i-1, k] + P[k]
    if debug:
        print DP.T

    return DP[-1, -1]


def fixed_len_subsequences(s, t, p=2):
    """
    Compute the fixed lenght subsequences kernel. This is a particular case
    of gap weighted subsequences kernel, for lambda = 1.
    Input:
        s is a string.
        t is a string.
        p is the subsequence size.
    Returns:
        The value of fixed lenght subsequences kernel between s and t.

    doctest:
    >>> fixed_len_subsequences("a", "b", p=0)
    1.0
    >>> fixed_len_subsequences("gatta", "cata", p=0)
    1.0
    >>> fixed_len_subsequences("", "cat")
    0.0
    >>> fixed_len_subsequences("cat", "")
    0.0
    >>> fixed_len_subsequences("gatta", "cata", p=1)
    6.0
    >>> fixed_len_subsequences("gatta", "cata", p=2)
    5.0
    >>> fixed_len_subsequences("gatta", "cata", p=3)
    2.0
    >>> fixed_len_subsequences(1, "cat")
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (fixed_len_subsequences): invalid input \
parameters.
    >>> fixed_len_subsequences("cat", 1.0)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (fixed_len_subsequences): invalid input \
parameters.
    >>> fixed_len_subsequences("cat", "cat", 1.0)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (fixed_len_subsequences): invalid input \
parameters.
    >>> fixed_len_subsequences("cat", "cat", -1)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (fixed_len_subsequences): invalid input \
parameters.
    >>> fixed_len_subsequences("cat", "cat", 1.0)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (fixed_len_subsequences): invalid input \
parameters.
    >>> fixed_len_subsequences("cat", "ca", 3)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (fixed_len_subsequences): len(s) or len(t) < p.
    >>> fixed_len_subsequences("ca", "cat", 3)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (fixed_len_subsequences): len(s) or len(t) < p.
    """
    assert (type(s) is str) and (type(t) is str) and (type(p) is int) and \
        (p >= 0), "*** ERROR (fixed_len_subsequences): invalid input \
parameters."
    if p == 0:
        return 1.0
    if ((s == "") or (t == "")):
        return 0.0
    assert (len(s) >= p) and (len(t) >= p), \
        "*** ERROR (fixed_len_subsequences): len(s) or len(t) < p."
    return gap_weighted_subsequences(s, t, p, lamb=1)


def gap_weighted_subsequences(s, t, p=2, lamb=0.5):
    """
    Compute the gap weighted subsequences kernel. This is a port from
    http://www.kernel-methods.net/matlab/manju/ssk_fast.m
    Input:
        s is a string.
        t is a string.
        p is the subsequence size.
        lamb is the weight parameter.
    Returns:
        The value of gap weighted subsequences kernel between s and t, with
    lamb weight parameter.

    doctest:
    >>> gap_weighted_subsequences("a", "b", p=0)
    1.0
    >>> gap_weighted_subsequences("gatta", "cata", p=0)
    1.0
    >>> gap_weighted_subsequences("", "cat")
    0.0
    >>> gap_weighted_subsequences("cat", "")
    0.0
    >>> gap_weighted_subsequences("gatta", "cata", p=1)
    1.5
    >>> gap_weighted_subsequences("gatta", "cata", p=2)
    0.1953125
    >>> gap_weighted_subsequences("gatta", "cata", p=3)
    0.015625
    >>> gap_weighted_subsequences("aqwsderf", "aqswdefr", p=3, lamb=1)
    44.0
    >>> gap_weighted_subsequences("aqwsderf", "aqswdefr", p=3, lamb=0.5)
    0.07061767578125
    >>> gap_weighted_subsequences("aqwsderf", "aqswdefr", p=2, lamb=0.25)
    0.01405915804207325
    >>> gap_weighted_subsequences(1, "cat")
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (gap_weighted_subsequences): invalid input \
parameters.
    >>> gap_weighted_subsequences("cat", 1.0)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (gap_weighted_subsequences): invalid input \
parameters.
    >>> gap_weighted_subsequences("cat", "cat", 1.0)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (gap_weighted_subsequences): invalid input \
parameters.
    >>> gap_weighted_subsequences("cat", "cat", -1)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (gap_weighted_subsequences): invalid input \
parameters.
    >>> gap_weighted_subsequences("cat", "cat", 1.0)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (gap_weighted_subsequences): invalid input \
parameters.
    >>> gap_weighted_subsequences("cat", "ca", 3)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (gap_weighted_subsequences): len(s) or len(t) \
< p.
    >>> gap_weighted_subsequences("ca", "cat", 3)
    Traceback (most recent call last):
        ...
    AssertionError: *** ERROR (gap_weighted_subsequences): len(s) or len(t) \
< p.
    """
    assert (type(s) is str) and (type(t) is str) and (type(p) is int) and \
        (p >= 0), "*** ERROR (gap_weighted_subsequences): invalid input \
parameters."
    if p == 0:
        return 1.0
    if ((s == "") or (t == "")):
        return 0.0
    assert (len(s) >= p) and (len(t) >= p), \
        "*** ERROR (gap_weighted_subsequences): len(s) or len(t) < p."
    n = len(s)
    m = len(t)
    K = np.tile(-1., (n, m))
    K_prime = np.tile(-1., (n, m, p))
    K[n-1, m-1], K_prime = ssk_fast_kernel(s, t, K, K_prime, p, lamb)

    return K[n-1, m-1]


def ssk_fast_kernel(sa, t, K, K_prime, p, lamb):
    """
    Gap weighted subsequences kernel auxiliar function.
    """
    n = len(sa)
    m = len(t)
    s = sa[:n-1]
    length_s = len(s)
    ans = 0
    if (len(s) < p) or (len(t) < p):
        ans = 0
    elif (K[len(s)-1, len(t)-1] == -1.):
        ans = ssk_fast_kernel(s, t, K, K_prime, p, lamb)[0]
    else:
        ans = K[len(s)-1, len(t)-1]
    letter = sa[n-1]
    pos_array = np.array([id for id in range(m) if t[id] == letter])
    for index in range(len(pos_array)):
        i = pos_array[index]
        length_t = len(t[:i])
        if (p-1) == 0:
            result = 1
        elif (length_s < (p-1)) or (length_t < (p-1)):
            result = 0
        elif K_prime[length_s-1, length_t-1, (p-1)-1] == -1.:
            result, K_prime = suffix_kernel(s, t[:i], K_prime, p-1, lamb)
        else:
            result = K_prime[length_s-1, length_t-1, p-1]
        ans += result * lamb**2

    return (ans, K_prime)


def suffix_kernel(sa, t, K_prime, i, lamb):
    """
    Gap weighted subsequences kernel auxiliar function.
    """
    n = len(sa)
    m = len(t)
    s = sa[:n-1]
    length_s = len(s)
    ans = 0
    if (len(s) < i) or (len(t) < i):
        ans = 0
    elif K_prime[len(s)-1, len(t)-1, 0] == -1.:
        ans = lamb * suffix_kernel(s, t, K_prime, i, lamb)[0]
    else:
        ans = lamb * K_prime[len(s)-1, len(t)-1, 0]
    letter = sa[n-1]
    pos_array = np.array([id for id in range(m) if t[id] == letter])
    for index in range(len(pos_array)):
        j = pos_array[index]
        length_t = len(t[:j])
        if (i-1) == 0:
            result = 1
        elif (length_s < (i-1)) or (length_t < (i-1)):
            result = 0
        elif K_prime[length_s-1, length_t-1, i-1] == -1.:
            result, K_prime = suffix_kernel(s, t[:j], K_prime, i-1, lamb)
        else:
            result = K_prime[length_s-1, length_t-1, i-1]
        ans += result * (lamb**(m-(j+1)+2))

    return (ans, K_prime)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
