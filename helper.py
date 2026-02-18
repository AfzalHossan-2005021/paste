import ot
import scipy
import torch
import numpy as np
from tqdm import tqdm
from typing import List
from anndata import AnnData

from ot.lp import emd
from ot.optim import line_search_armijo
from ot.utils import list_to_array, get_backend


def generic_conditional_gradient_incent(a, b, M1, M2, f, df, reg1, reg2, lp_solver, line_search,
                                         gamma, G0=None, numItermax=6000, stopThr=1e-9,
                                         stopThr2=1e-9, verbose=False, log=False, **kwargs):
    r"""
    Solve the general regularized OT problem or its semi-relaxed version with
    conditional gradient or generalized conditional gradient depending on the
    provided linear program solver.

        The function solves the following optimization problem if set as a conditional gradient:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_1} \cdot f(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b} (optional constraint)

             \gamma &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`f` is the regularization term (and `df` is its gradient)
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)

    The algorithm used for solving the problem is conditional gradient as discussed in :ref:`[1] <references-cg>`

        The function solves the following optimization problem if set a generalized conditional gradient:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_1}\cdot f(\gamma) + \mathrm{reg_2}\cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0

    where :

    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`

    The algorithm used for solving the problem is the generalized conditional gradient as discussed in :ref:`[5, 7] <references-gcg>`

    Parameters
    ----------
    a : array-like, shape (ns,)
        samples weights in the source domain
    b : array-like, shape (nt,)
        samples weights in the target domain

    a: initial distribution(uniform) of sliceA spots
    b: initial distribution(uniform) of sliceB spots

    M1: cosine dist of gene expression matrices of two slices
    M2: jensenshannon dist of niche of two slices
    f : function
        Regularization function taking a transportation matrix as argument
    df: function
        Gradient of the regularization function taking a transportation matrix as argument
    reg1 : float
        Regularization term >0
    reg2 : float,
        Entropic Regularization term >0. Ignored if set to None.
    lp_solver: function,
        linear program solver for direction finding of the (generalized) conditional gradient.
        If set to emd will solve the general regularized OT problem using cg.
        If set to lp_semi_relaxed_OT will solve the general regularized semi-relaxed OT problem using cg.
        If set to sinkhorn will solve the general regularized OT problem using generalized cg.
    line_search: function,
        Function to find the optimal step. Currently used instances are:
        line_search_armijo (generic solver). solve_gromov_linesearch for (F)GW problem.
        solve_semirelaxed_gromov_linesearch for sr(F)GW problem. gcg_linesearch for the Generalized cg.
    G0 :  array-like, shape (ns,nt), optional
        initial guess (default is indep joint density)
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on the relative variation (>0)
    stopThr2 : float, optional
        Stop threshold on the absolute variation (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Added by Anup Bhowmik
    ------------------------
    gamma: float, optional
        weight of the second regularization term (default is 0.5)
    --------------------------


    **kwargs : dict
             Parameters for linesearch

    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-cg:
    .. _references_gcg:
    References
    ----------

    .. [1] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014). Regularized discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.

    .. [5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy, "Optimal Transport for Domain Adaptation," in IEEE Transactions on Pattern Analysis and Machine Intelligence , vol.PP, no.99, pp.1-1

    .. [7] Rakotomamonjy, A., Flamary, R., & Courty, N. (2015). Generalized conditional gradient: analysis of convergence and applications. arXiv preprint arXiv:1510.06567.

    See Also
    --------
    ot.lp.emd : Unregularized optimal transport
    ot.bregman.sinkhorn : Entropic regularized optimal transport
    """

    # new code starts
    a, b, M1, M2, G0 = list_to_array(a, b, M1, M2, G0)
    if isinstance(M1, int) or isinstance(M1, float):
        nx = get_backend(a, b)
    else:
        nx = get_backend(a, b, M1)

    if isinstance(M2, int) or isinstance(M2, float):
        nx = get_backend(a, b)
    else:
        nx = get_backend(a, b, M2)

    # new code ends

    loop = 1

    if log:
        log = {'loss': []}

    if G0 is None:
        # G0 is kept None by default
        
        G2 = nx.outer(a, b)
        # make G uniform distribution matrix of size (ns, nt)
        G1 = nx.ones((a.shape[0], b.shape[0])) / (a.shape[0] * b.shape[0])

        # todo: integrate the cell-type aware initialization


        G = G1
        # print the shape of G
        # print("G shape: ", G.shape)
    else:
        # to not change G0 in place.
        G = nx.copy(G0)

    def cost(G):
        alpha = reg1
        
        # with niche aware
        return (1-alpha) * (nx.sum(M1 * G) + gamma * nx.sum(M2 * G)) + alpha * f(G)

        # without niche aware
        # return (1-alpha) * (nx.sum(M1 * G)) + alpha * f(G)

    

    cost_G = cost(G)
    if log:
        log['loss'].append(cost_G)

    it = 0

    if verbose:
        print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
            'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
        print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, cost_G, 0, 0))

    while loop:

        it += 1
        old_cost_G = cost_G
        # problem linearization
        # gradient descent
        Mi = M1 + reg1 * df(G)

        if not (reg2 is None):
            Mi = Mi + reg2 * (1 + nx.log(G))
        # set M positive
        Mi = Mi + nx.min(Mi)

        # solve linear program
        Gc, innerlog_ = lp_solver(a, b, Mi, **kwargs)

        # line search
        deltaG = Gc - G

        alpha, fc, cost_G = line_search(cost, G, deltaG, Mi, cost_G, **kwargs)

        G = G + alpha * deltaG

        # test convergence
        if it >= numItermax:
            loop = 0

        abs_delta_cost_G = abs(cost_G - old_cost_G)
        relative_delta_cost_G = abs_delta_cost_G / abs(cost_G)
        if relative_delta_cost_G < stopThr or abs_delta_cost_G < stopThr2:
            loop = 0

        if log:
            log['loss'].append(cost_G)

        if verbose:
            if it % 20 == 0:
                print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
                    'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
            print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, cost_G, relative_delta_cost_G, abs_delta_cost_G))

    if log:
        log.update(innerlog_)
        return G, log
    else:
        return G


def cg_incent(a, b, M1, M2, reg, f, df, gamma, G0=None, line_search=line_search_armijo,
       numItermax=6000, numItermaxEmd=100000, stopThr=1e-9, stopThr2=1e-9,
       verbose=False, log=False, **kwargs):
    r"""
    Solve the general regularized OT problem with conditional gradient

        The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg} \cdot f(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`f` is the regularization term (and `df` is its gradient)
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)

    The algorithm used for solving the problem is conditional gradient as discussed in :ref:`[1] <references-cg>`


    Parameters
    ----------
    # a : array-like, shape (ns,)
    #     samples weights in the source domain
    # b : array-like, shape (nt,)
    #     samples in the target domain

    # a: initial distribution(uniform) of sliceA spots
    # b: initial distribution(uniform) of sliceB spots
    
    # M : array-like, shape (ns, nt)
    #     loss matrix

    # M1: cosine dist of gene expression matrices of two slices
    # M2: jensenshannon dist of niche of two slices

    
    reg : float
        Regularization term >0
    G0 :  array-like, shape (ns,nt), optional
        initial guess (default is indep joint density)
    line_search: function,
        Function to find the optimal step.
        Default is line_search_armijo.
    numItermax : int, optional
        Max number of iterations
    numItermaxEmd : int, optional
        Max number of iterations for emd
    stopThr : float, optional
        Stop threshold on the relative variation (>0)
    stopThr2 : float, optional
        Stop threshold on the absolute variation (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    **kwargs : dict
             Parameters for linesearch

    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-cg:
    References
    ----------

    .. [1] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014). Regularized discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.

    See Also
    --------
    ot.lp.emd : Unregularized optimal transport
    ot.bregman.sinkhorn : Entropic regularized optimal transport

    """

    def lp_solver(a, b, M, **kwargs):
        return emd(a, b, M, numItermaxEmd, log=True)

    return generic_conditional_gradient_incent(a, b, M1, M2, f, df, reg, None, lp_solver, line_search, G0=G0,
                                               gamma = gamma, numItermax=numItermax, stopThr=stopThr,
                                               stopThr2=stopThr2, verbose=verbose, log=log, **kwargs)


def kl_divergence_corresponding_backend(X, Y):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.

    Takes advantage of POT backend to speed up computation.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    nx = ot.backend.get_backend(X,Y)

    X = X/nx.sum(X,axis=1, keepdims=True)
    Y = Y/nx.sum(Y,axis=1, keepdims=True)
    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum('ij,ij->i',X,log_X)
    X_log_X = nx.reshape(X_log_X,(1,X_log_X.shape[0]))

    X_log_Y = nx.einsum('ij,ij->i',X,log_Y)
    X_log_Y = nx.reshape(X_log_Y,(1,X_log_Y.shape[0]))
    D = X_log_X.T - X_log_Y.T
    return nx.to_numpy(D)

def jensenshannon_distance_1_vs_many_backend(X, Y):
    """
    Returns pairwise Jensenshannon distance (over all pairs of samples) of two matrices X and Y.

    Takes advantage of POT backend to speed up computation.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    assert X.shape[0] == 1
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nx = ot.backend.get_backend(X,Y)        # np or torch depending upon gpu availability
    X = nx.concatenate([X] * Y.shape[0], axis=0) # broadcast X
    X = X/nx.sum(X,axis=1, keepdims=True)   # normalize
    Y = Y/nx.sum(Y,axis=1, keepdims=True)   # normalize
    M = (X + Y) / 2.0
    kl_X_M = torch.from_numpy(kl_divergence_corresponding_backend(X, M))
    kl_Y_M = torch.from_numpy(kl_divergence_corresponding_backend(Y, M))
    js_dist = nx.sqrt((kl_X_M + kl_Y_M) / 2.0).T[0]
    return js_dist

def jensenshannon_divergence_backend(X, Y, *, verbose: bool = True):
    """
    This function is added ny Nuwaisir
    
    Returns pairwise JS divergence (over all pairs of samples) of two matrices X and Y.

    Takes advantage of POT backend to speed up computation.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    if verbose:
        print("Calculating cost matrix")

    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    nx = ot.backend.get_backend(X,Y)
    # nx = ot.backend.NumpyBackend()

    # X = X.cpu().detach().numpy()
    # Y = Y.cpu().detach().numpy()

    # print(nx.unique(nx.isnan(X)))
    # print(nx.unique(nx.isnan(Y)))
        
    
    X = X/nx.sum(X,axis=1, keepdims=True)
    Y = Y/nx.sum(Y,axis=1, keepdims=True)

    n = X.shape[0]
    m = Y.shape[0]
    
    js_dist = nx.zeros((n, m))

    for i in tqdm(range(n), disable=not verbose):
        js_dist[i, :] = jensenshannon_distance_1_vs_many_backend(X[i:i+1], Y)


    # if("numpy" in str(type(js_dist))):
    #     js_dist = torch.from_numpy(js_dist)
        
    if verbose:
        print("Finished calculating cost matrix")
    # print(nx.unique(nx.isnan(js_dist)))

    if torch.cuda.is_available():
        try:
            return js_dist.numpy()
        except:
            return js_dist
    else:
        return js_dist
    
    # print("type of js dist:" , type(js_dist))
    # return js_dist
    
    # print("vectorized jsd")
    # X = X/nx.sum(X,axis=1, keepdims=True)
    # Y = Y/nx.sum(Y,axis=1, keepdims=True)

    # mid = (X[:, None] + Y) / 2
    # n = X.shape[0]
    # m = Y.shape[0]
    # d = X.shape[1]
    # l = nx.ones((n, m, d)) * X[:, None, :]
    # r = nx.ones((n, m, d)) * Y[None, :, :]
    # l_2d = nx.reshape(l, (-1, l.shape[2]))
    # r_2d = nx.reshape(r, (-1, r.shape[2]))
    # m_2d = (l_2d + r_2d) / 2.0
    # kl_l_m = kl_divergence_corresponding_backend(l_2d, m_2d)
    # kl_r_m = kl_divergence_corresponding_backend(r_2d, m_2d)

    # js_dist = nx.sqrt((kl_l_m + kl_r_m) / 2.0)
    # return nx.reshape(js_dist, (n, m))


def filter_for_common_genes(
    slices: List[AnnData]) -> None:
    """
    Filters for the intersection of genes between all slices.

    Args:
        slices: List of slices.
    """
    assert len(slices) > 0, "Cannot have empty list."

    common_genes = slices[0].var.index
    for s in slices:
        common_genes = intersect(common_genes, s.var.index)
    for i in range(len(slices)):
        slices[i] = slices[i][:, common_genes]
    print('Filtered all slices for common genes. There are ' + str(len(common_genes)) + ' common genes.')

def match_spots_using_spatial_heuristic(
    X,
    Y,
    use_ot: bool = True) -> np.ndarray:
    """
    Calculates and returns a mapping of spots using a spatial heuristic.

    Args:
        X (array-like, optional): Coordinates for spots X.
        Y (array-like, optional): Coordinates for spots Y.
        use_ot: If ``True``, use optimal transport ``ot.emd()`` to calculate mapping. Otherwise, use Scipy's ``min_weight_full_bipartite_matching()`` algorithm.

    Returns:
        Mapping of spots using a spatial heuristic.
    """
    n1,n2=len(X),len(Y)
    X,Y = norm_and_center_coordinates(X),norm_and_center_coordinates(Y)
    dist = scipy.spatial.distance_matrix(X,Y)
    if use_ot:
        pi = ot.emd(np.ones(n1)/n1, np.ones(n2)/n2, dist)
    else:
        row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(scipy.sparse.csr_matrix(dist))
        pi = np.zeros((n1,n2))
        pi[row_ind, col_ind] = 1/max(n1,n2)
        if n1<n2: pi[:, [(j not in col_ind) for j in range(n2)]] = 1/(n1*n2)
        elif n2<n1: pi[[(i not in row_ind) for i in range(n1)], :] = 1/(n1*n2)
    return pi

def kl_divergence(X, Y):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    X = X/X.sum(axis=1, keepdims=True)
    Y = Y/Y.sum(axis=1, keepdims=True)
    log_X = np.log(X)
    log_Y = np.log(Y)
    X_log_X = np.matrix([np.dot(X[i],log_X[i].T) for i in range(X.shape[0])])
    D = X_log_X.T - np.dot(X,log_Y.T)
    return np.asarray(D)

def kl_divergence_backend(X, Y):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.

    Takes advantage of POT backend to speed up computation.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    nx = ot.backend.get_backend(X,Y)

    X = X/nx.sum(X,axis=1, keepdims=True)
    Y = Y/nx.sum(Y,axis=1, keepdims=True)

    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum('ij,ij->i',X,log_X)
    X_log_X = nx.reshape(X_log_X,(1,X_log_X.shape[0]))
    D = X_log_X.T - nx.dot(X,log_Y.T)
    return nx.to_numpy(D)


def intersect(lst1, lst2):
    """
    Gets and returns intersection of two lists.

    Args:
        lst1: List
        lst2: List

    Returns:
        lst3: List of common elements.
    """

    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

def norm_and_center_coordinates(X):
    """
    Normalizes and centers coordinates at the origin.

    Args:
        X: Numpy array

    Returns:
        X_new: Updated coordiantes.
    """
    return (X-X.mean(axis=0))/min(scipy.spatial.distance.pdist(X))

def apply_trsf(
    M: np.ndarray,
    translation: List[float],
    points: np.ndarray) -> np.ndarray:
    """
    Apply a rotation from a 2x2 rotation matrix `M` together with
    a translation from a translation vector of length 2 `translation` to a list of
    `points`.

    Args:
        M (nd.array): A 2x2 rotation matrix.
        translation (nd.array): A translation vector of length 2.
        points (nd.array): A nx2 array of `n` points 2D positions.

    Returns:
        (nd.array) A nx2 matrix of the `n` points transformed.
    """
    if not isinstance(translation, np.ndarray):
        translation = np.array(translation)
    trsf = np.identity(3)
    trsf[:-1, :-1] = M
    tr = np.identity(3)
    tr[:-1, -1] = -translation
    trsf = trsf @ tr

    flo = points.T
    flo_pad = np.pad(flo, ((0, 1), (0, 0)), constant_values=1)
    return ((trsf @ flo_pad)[:-1]).T


from sklearn.metrics.pairwise import euclidean_distances
def get_neighborhood_distribution(curr_slice, radius, *, verbose: bool = True):
    """
    This method is added by Anup Bhowmik
    Args:
        curr_slice: Slice to get niche distribution for.
        pairwise_distances: Pairwise distances between cells of a slice.
        radius: Radius of the niche.

    Returns:
        niche_distribution: Niche distribution for the slice.
    """

    # print ("radius", radius)

    unique_cell_types = np.array(list(curr_slice.obs['cell_type_annot'].unique()))
    cell_type_to_index = dict(zip(unique_cell_types, list(range(len(unique_cell_types)))))
    cells_within_radius = np.zeros((curr_slice.shape[0], len(unique_cell_types)), dtype=float)

    # print("time taken for cell type", time_cell_type_end-time_cell_type_start)

    source_coords = curr_slice.obsm['spatial']
    distances = euclidean_distances(source_coords, source_coords)

    for i in tqdm(range(curr_slice.shape[0]), disable=not verbose):
        # find the indices of the cells within the radius

        target_indices = np.where(distances[i] <= radius)[0]
        # print("i", i)
        # print(target_indices)

        for ind in target_indices:
            cell_type_str_j = str(curr_slice.obs['cell_type_annot'][ind])
            cells_within_radius[i][cell_type_to_index[cell_type_str_j]] += 1

    return np.array(cells_within_radius)

def cosine_dist_calculator(
    sliceA,
    sliceB,
    sliceA_name,
    sliceB_name,
    filePath,
    use_rep=None,
    use_gpu=False,
    nx=ot.backend.NumpyBackend(),
    beta=0.8,
    overwrite=False,
    *,
    verbose: bool = True,
):
    from sklearn.metrics.pairwise import cosine_distances
    import os
    import pandas as pd

    A_X, B_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceA,use_rep))), nx.from_numpy(to_dense_array(extract_data_matrix(sliceB,use_rep)))

    if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
        A_X = A_X.cuda()
        B_X = B_X.cuda()

   
    s_A = A_X + 0.01
    s_B = B_X + 0.01

    
    one_hot_cell_type_sliceA = pd.get_dummies(sliceA.obs['cell_type_annot'])
    # print ("one_hot_cell_type_sliceA type: ", type(one_hot_cell_type_sliceA))
    one_hot_cell_type_sliceA = one_hot_cell_type_sliceA.to_numpy()

    one_hot_cell_type_sliceB = pd.get_dummies(sliceB.obs['cell_type_annot'])
    one_hot_cell_type_sliceB = one_hot_cell_type_sliceB.to_numpy()

    if isinstance(nx,ot.backend.TorchBackend):
        s_A = s_A.cpu().detach().numpy()
        s_B = s_B.cpu().detach().numpy()

    # Concatenate along a specified axis (0 for rows, 1 for columns)
    s_A = np.concatenate((s_A, beta * one_hot_cell_type_sliceA), axis=1)
    s_B = np.concatenate((s_B, beta * one_hot_cell_type_sliceB), axis=1)

    s_A = torch.from_numpy(s_A)
    s_B = torch.from_numpy(s_B)

    if torch.cuda.is_available():
        if verbose:
            print("CUDA is available on your system.")
        s_A = s_A.to('cuda')
        s_B = s_B.to('cuda')

    else:
        if verbose:
            print("CUDA is not available on your system.")

    fileName = f"{filePath}/cosine_dist_gene_expr_{sliceA_name}_{sliceB_name}.npy"
    
    if os.path.exists(fileName) and not overwrite:
        if verbose:
            print("Loading precomputed Cosine distance of gene expression for slice A and slice B")
        cosine_dist_gene_expr = np.load(fileName)
    else:
        if verbose:
            print("Calculating cosine dist of gene expression for slice A and slice B")

        # calculate cosine distance manually
        # cosine_dist_gene_expr = 1 - (s_A @ s_B.T) / s_A.norm(dim=1)[:, None] / s_B.norm(dim=1)[None, :]
        # cosine_dist_gene_expr = cosine_dist_gene_expr.cpu().detach().numpy()

        # use sklearn's cosine_distances
        if torch.cuda.is_available():
            s_A = s_A.cpu().detach().numpy()
            s_B = s_B.cpu().detach().numpy()
        cosine_dist_gene_expr = cosine_distances(s_A, s_B)

        print("Saving cosine dist of gene expression for slice A and slice B")
        np.save(fileName, cosine_dist_gene_expr)

    return cosine_dist_gene_expr

def weighted_distance_calculator(sliceA, sliceB, sliceA_name, sliceB_name, filePath, 
                                 use_rep=None, use_gpu=False, nx=ot.backend.NumpyBackend(), 
                                 beta=0.8, overwrite=False, distance_metric='cosine'):
    """
    Improved distance calculator that separately computes gene expression and cell type 
    distances, then combines them with a weighted sum. This approach is more biologically 
    meaningful than concatenating features.
    
    This method computes:
        D_combined = (1 - beta) * D_gene + beta * D_celltype
    
    where:
        - D_gene: pairwise distance based on gene expression (cosine or euclidean)
        - D_celltype: binary distance (0 if same cell type, 1 if different)
        - beta: weight controlling importance of cell type vs gene expression
    
    Args:
        sliceA: First AnnData slice with gene expression and cell type annotations.
        sliceB: Second AnnData slice with gene expression and cell type annotations.
        sliceA_name: Name identifier for slice A (used for caching).
        sliceB_name: Name identifier for slice B (used for caching).
        filePath: Directory path for saving/loading cached results.
        use_rep: If None, uses slice.X for gene expression. Otherwise uses slice.obsm[use_rep].
        use_gpu: If True, use GPU for computation (requires CUDA). Default: False.
        nx: POT backend (NumpyBackend or TorchBackend). Default: NumpyBackend.
        beta: Weight parameter (0 to 1). 
              - beta=0: pure gene expression distance
              - beta=1: pure cell type distance
              - beta=0.5: equal weighting
              Default: 0.8
        overwrite: If True, recompute even if cached file exists. Default: False.
        distance_metric: Distance metric for gene expression. Options: 'cosine' or 'euclidean'.
                        Default: 'cosine'.
    
    Returns:
        D_combined: numpy array of shape (n_cells_A, n_cells_B) containing pairwise distances.
    
    Notes:
        - Requires 'cell_type_annot' column in both sliceA.obs and sliceB.obs
        - Results are cached as .npy files for faster subsequent loading
        - More interpretable than feature concatenation: each modality uses appropriate metric
        - No scale mixing issues between gene expression and binary cell type features
        
    Example:
        >>> dist = weighted_distance_calculator(slice1, slice2, "4wk", "24wk", 
        ...                                     "./cache", beta=0.7)
        >>> # 70% weight on cell type, 30% on gene expression
    """
    from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
    import os
    
    fileName = f"{filePath}/weighted_dist_{distance_metric}_beta{beta}_{sliceA_name}_{sliceB_name}.npy"
    
    if os.path.exists(fileName) and not overwrite:
        print(f"Loading precomputed weighted distance ({distance_metric}, beta={beta}) for slice A and slice B")
        return np.load(fileName)
    
    print(f"Calculating weighted distance ({distance_metric}, beta={beta}) for slice A and slice B")
    
    # Extract gene expression matrices
    A_X = to_dense_array(extract_data_matrix(sliceA, use_rep))
    B_X = to_dense_array(extract_data_matrix(sliceB, use_rep))
    
    # Add small constant to avoid numerical issues
    A_X = A_X + 0.01
    B_X = B_X + 0.01
    
    # Convert to appropriate backend if using GPU
    if isinstance(nx, ot.backend.TorchBackend) and use_gpu:
        A_X = torch.from_numpy(A_X).cuda()
        B_X = torch.from_numpy(B_X).cuda()
    
    # Compute gene expression distance
    if torch.cuda.is_available() and use_gpu and isinstance(A_X, torch.Tensor):
        A_X_np = A_X.cpu().detach().numpy()
        B_X_np = B_X.cpu().detach().numpy()
    else:
        A_X_np = A_X if isinstance(A_X, np.ndarray) else A_X
        B_X_np = B_X if isinstance(B_X, np.ndarray) else B_X
    
    if distance_metric == 'cosine':
        D_gene = cosine_distances(A_X_np, B_X_np)
    elif distance_metric == 'euclidean':
        D_gene = euclidean_distances(A_X_np, B_X_np)
        # Normalize euclidean distances to [0, 1] range for consistent weighting
        if D_gene.max() > 0:
            D_gene = D_gene / D_gene.max()
    else:
        raise ValueError(f"Unknown distance_metric: {distance_metric}. Use 'cosine' or 'euclidean'.")
    
    # Compute cell type distance (binary: 0 if same, 1 if different)
    cell_types_A = sliceA.obs['cell_type_annot'].values
    cell_types_B = sliceB.obs['cell_type_annot'].values
    
    # Broadcasting to create pairwise comparison matrix
    D_celltype = (cell_types_A[:, None] != cell_types_B[None, :]).astype(float)
    
    # Weighted combination
    # (1 - beta) weight on gene expression, beta weight on cell type
    D_combined = (1 - beta) * D_gene + beta * D_celltype
    
    # Save result
    print(f"Saving weighted distance for slice A and slice B")
    np.save(fileName, D_combined)
    
    return D_combined

def pairwise_msd(A, B):
    A = np.asarray(A)
    B = np.asarray(B)

    # A: (m, d), B: (n, d)
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]  # shape: (m, n, d)
    msd = np.mean(diff ** 2, axis=2)  # shape: (m, n)
    return msd

## Covert a sparse matrix into a dense np array
to_dense_array = lambda X: X.toarray() if isinstance(X,scipy.sparse.csr.spmatrix) else np.array(X)

## Returns the data matrix or representation
extract_data_matrix = lambda adata,rep: adata.X if rep is None else adata.obsm[rep]