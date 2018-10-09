import numpy as np

def tensordotmat(ts, mat, axes_ts=1, axes_mt=1):
    """
    Compute the tensor dot product with matrix (2D).
    ts belongs to the space of I1*I2*I3*...*Ij*...*IN, (I is integer value)
    mat is a matrix with the shape of Pj*Ij.
    This function help to implement the ts *_j mat
    ts *_j mat = I1*I2*I3*...*Pj*...*IN
    Parameters
    ----------
    ts, mat : array_like, len(shape) >= 1
        Tensors to "dot".
    axes_ts: variable type
        * integer
    axes_mt: 
        * integer
    requirement:
        the axes_ts shape of ts must be same as axes_mt shape of mat,
        in another word, the Ij of ts and Ij of mat must be equal.
    Return
    ----------
    a new tensor with the shape of I1*I2*I3*...*Pj*...*IN
    See Also
    --------
    dot, einsum, tensordot
    Note
    --------
    This function derives from tensordot, but aim at the dot product between mat and tensor.and

    Examples
     --------
    A "traditional" example:   
    >>> arr = np.array([[[1, 1, 1],
                [0, 0, 0],
                [2, 2, 2]],

               [[0, 0, 0],
                [4, 4, 4],
                [0, 0, 0]]])

    >>> w = np.diag([1, 1, 1])
    >>> tensordotmat(arr, w, axes_ts=1,axes_mt=1)
    array([[[1, 1, 1],
        [0, 0, 0],
        [2, 2, 2]],

       [[0, 0, 0],
        [4, 4, 4],
        [0, 0, 0]]])
    >>> # And the order of dimension does not change after tensordotmat
    >>> # When you try tensordot, the order can be changed.
    >>> w = [[1,0,0],[0,1,0]]
    >>> tensordotmat(arr, w, axes_ts=1,axes_mt=1)
    array([[[1, 1, 1],
        [0, 0, 0]],

       [[0, 0, 0],
        [4, 4, 4]]])
    """
    try:
        na = len(axes_ts)
        axes_ts = list(axes_ts)
    except TypeError:
        axes_ts = [axes_ts]
        na = 1
    try:
        nb = len(axes_mt)
        axes_mt = list(axes_mt)
    except TypeError:
        axes_mt = [axes_mt]
        nb = 1

    ts, mat = np.asarray(ts), np.asarray(mat)
    as_ = ts.shape
    nda = len(ts.shape)
    bs = mat.shape
    ndb = len(mat.shape)
    equal = True
    if (na != nb): equal = False
    else:
        for k in range(na):
            if as_[axes_ts[k]] != bs[axes_mt[k]]:
                equal = False
                break
            if axes_ts[k] < 0:
                axes_ts[k] += nda
            if axes_mt[k] < 0:
                axes_mt[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    # Move the axes to sum over to the end of "ts"
    # and to the front of "mat"
    notin = [k for k in range(nda) if k not in axes_ts]
    newaxes_a = notin + axes_ts
    N2 = 1
    for axis in axes_ts:
        N2 *= as_[axis]
    newshape_a = (-1, N2)
    olda = [as_[axis] for axis in notin]
    
    notin = [k for k in range(ndb) if k not in axes_mt]
    newaxes_b = axes_mt + notin
    N2 = 1
    for axis in axes_mt:
        N2 *= bs[axis]
    newshape_b = (N2, -1)
    
    oldb = bs[0]
    at = ts.transpose(newaxes_a).reshape(newshape_a)
    bt = mat.transpose(newaxes_b).reshape(newshape_b)
    res = np.dot(at, bt)
    
    res = res.reshape(olda+[oldb])
    baseTs = range(nda-1)
    baseTs.insert(axes_ts[0], nda-1)
    return res.transpose(baseTs)
    