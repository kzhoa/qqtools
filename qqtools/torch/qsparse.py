import numpy as np
import scipy
import torch
from scipy import sparse as sp

"""
matrix type convert
"""


def sparse_to_tuple(mx):
    """Convert scipy sparse matrix to tuple (coords,values,shape)."""
    if not sp.isspmatrix_coo(mx):
        mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    values = mx.data
    shape = mx.shape
    return coords, values, shape


def scipy2csrtensor(A):
    if isinstance(A, (np.ndarray, np.matrix)):
        A = scipy.sparse.csr_matrix(A)
    elif scipy.sparse.issparse(A):
        A = A.tocsr()
    else:
        raise TypeError(f"TypeError: input matrix should be scipy.sparse, but got {type(A)}.")

    values = torch.from_numpy(A.data)
    indices = torch.LongTensor(A.indices)
    crow_indices = torch.LongTensor(A.indptr)
    csr_tensor = torch.sparse_csr_tensor(crow_indices, indices, values, size=A.shape).coalesce()
    return csr_tensor


def scipy2cootensor(A):
    """Convert scipy.sparse.coo_matrix to torch.sparse.FloatTensor"""
    if isinstance(A, (np.ndarray, np.matrix)):
        A = scipy.sparse.csr_matrix(A)
    elif scipy.sparse.issparse(A):
        A = A.tocoo()
    else:
        raise TypeError(f"TypeError: input matrix should be scipy.sparse, but got {type(A)}.")

    v = torch.from_numpy(A.data)
    i = torch.LongTensor(np.array([A.row, A.col]))

    coo_tensor = torch.sparse_coo_tensor(i, v, size=A.shape).coalesce()
    return coo_tensor


def scipy2sparsetensor(A):
    """for the compatibility of some old version codes"""
    if scipy.sparse.isspmatrix_coo(A):
        return scipy2cootensor(A)
    elif scipy.sparse.isspmatrix_csr(A):
        return scipy2csrtensor(A)
    elif scipy.sparse.isspmatrix_csc(A):
        return scipy2cootensor(A)  # 默认将csc转coo，因为torch没有csc的layout。
        raise NotImplementedError


def sparsetensor2scipy(A):
    """返回的scipy.sparse取决于torch.sparse.tensor的layout"""
    assert isinstance(A, torch.Tensor)
    # A = A.coalesce()
    if A.layout == torch.sparse_csr:
        vals = A.values().numpy()
        indices = A.col_indices().numpy()
        ind_ptr = A.crow_indices().numpy()
        sparse_matrix = scipy.sparse.csr_matrix((vals, indices, ind_ptr), shape=A.shape)
    elif A.layout == torch.sparse_coo:
        indices = A.indices()
        row, col = indices[0].numpy(), indices[1].numpy()
        vals = A.values().numpy()
        sparse_matrix = scipy.sparse.coo_matrix((vals, (row, col)), shape=A.shape)
    else:
        raise RuntimeError(f"qqtools.sparsetensor2scipy doesnt support: {A.layout}")

    return sparse_matrix


def to_coo(A):
    if scipy.sparse.issparse(A):
        return A.tocoo()
    elif isinstance(A, (np.ndarray, np.matrix)):
        return scipy.sparse.coo_matrix(A)

    elif isinstance(A, torch.Tensor):
        if A.layout == torch.sparse_coo:
            return A
        elif A.layout == torch.sparse_csr:
            csr_matrix = sparsetensor2scipy(A)
            coo_tensor = scipy2cootensor(csr_matrix)
            return coo_tensor
        else:
            # dense tensor
            return A.to_sparse()


def to_csr(A):
    if scipy.sparse.issparse(A):
        return A.tocsr()
    elif isinstance(A, (np.ndarray, np.matrix)):
        return scipy.sparse.csr_matrix(A)

    elif isinstance(A, torch.Tensor):
        if A.layout == torch.sparse_csr:
            return A
        else:
            # dense tensor or coo sparse tensor
            return A.to_sparse_csr()


def to_cootensor(A):
    if scipy.sparse.issparse(A):
        return scipy2cootensor(A)
    elif isinstance(A, (np.ndarray, np.matrix)):
        return torch.from_numpy(A).to_sparse()
    elif isinstance(A, torch.Tensor):
        return to_coo(A)
    else:
        raise TypeError(f"not supported type: {type(A)}")


def to_csrtensor(A):
    if scipy.sparse.issparse(A):
        return scipy2csrtensor(A)
    elif isinstance(A, (np.ndarray, np.matrix)):
        return scipy2csrtensor(A)
    elif isinstance(A, torch.Tensor):
        return to_csr(A)
    else:
        raise TypeError(f"not supported type: {type(A)}")


def to_coomatrix(A):
    if isinstance(A, torch.Tensor):
        if A.layout == torch.sparse_coo:
            return sparsetensor2scipy(A)
        elif A.layout == torch.sparse_csr:
            return sparsetensor2scipy(A).tocoo()
        else:
            # dense tensor
            return scipy.sparse.coo_matrix(A.numpy())
    elif scipy.sparse.issparse(A):
        return A.tocoo()
    elif isinstance(A, (np.ndarray, np.matrix)):
        return scipy.sparse.coo_matrix(A)
    else:
        raise TypeError(f"not supported type: {type(A)}")


def to_csrmatrix(A):
    if isinstance(A, torch.Tensor):
        if A.layout == torch.sparse_coo:
            return sparsetensor2scipy(A).tocsr()
        elif A.layout == torch.sparse_csr:
            return sparsetensor2scipy(A)
        else:
            # dense tensor
            return scipy.sparse.csr_matrix(A.numpy())
    elif scipy.sparse.issparse(A):
        return A.tocsr()
    elif isinstance(A, (np.ndarray, np.matrix)):
        return scipy.sparse.csr_matrix(A)
    else:
        raise TypeError(f"not supported type: {type(A)}")


def densetensor2sparsetensor(x):
    """converts dense tensor x to sparse format"""
    x_typename = torch.typename(x).split(".")[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


"""
sparse opetator
"""


def sparse_mean(x: torch.sparse.Tensor, dim):
    # input: torch.sparse
    # idx= x.indicese()
    # val = x.values()
    raise NotImplementedError


def eliminate_zeros(m: torch.sparse.Tensor):
    """TODO"""
    assert scipy.sparse.issparse(m)
    mask = m._values().nonzero()
    nv = m._values().index_select(0, mask.view(-1))
    ni = m._indices().index_select(1, mask.view(-1))
    return torch.sparse_coo_tensor(ni, nv, torch.Size([2, 3]), dtype=m.dtype)
