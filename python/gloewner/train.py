import numpy as np
from .barycentric import barycentricFunction
from .estimator import samplerEff
from .logging import logger

def getSample(sampler, z_test, idx, j, is_system_selfadjoint):
    z_sample = z_test.pop(idx) # remove sample point from test set
    y = samplerEff(sampler, z_sample)
    logger.info("{}: sampled at z={}j".format(j, z_sample))
    if is_system_selfadjoint:
        yC = y.conj()
    else:
        yC = samplerEff(sampler, -z_sample)
        logger.info("{}: sampled at z={}j".format(j, -z_sample))
    return z_sample, y, yC

def getBarycentricCoeffs(L):
    R = np.linalg.qr(L)[1]
    Vh = np.linalg.svd(R)[2]
    return Vh[-1].conj()

def trainSurrogate(sampler, z_test, start_z_idx, estimator, Smax, N_memory = 1,
                   return_estimate = True, is_system_selfadjoint = True):
    """Train rational model using the greedy Loewner framework
    Inputs:
    sampler: Callable to evaluate high-fidelity model. Note: model is
        evaluated at 1j * z.
    z_test: List of values of z among which to choose samples.
    start_z_idx: List of indices (within z_test) of starting samples. If empty,
        it is initialized to [0].
    estimator: Class of type estimator.estimator to drive the adaptivity.
    Smax: Maximum number of high-fidelity samples to be taken.
    N_memory: Depth of memory. Method terminates successfully only if
        estimator yields a "pass" N_memory times in a row.
    return_estimate: Whether to also return an error estimate.
    is_system_selfadjoint: Whether system that generates data (through
        sampler) is selfadjoint:
            sampler(conj(z)) == conj(sampler(z)) for all complex z.
        (In practice, it's enough for the relation to be true on the
        imaginary axis.) If True, saves half the high-fidelity samples.
    """
    z_test = list(z_test)
    if not hasattr(start_z_idx, "__len__"): start_z_idx = [start_z_idx]
    if not len(start_z_idx): start_z_idx = [0]
    start_z_idx = np.unique(start_z_idx)[::-1]

    is_system_selfadjoint = (is_system_selfadjoint
                         and not np.any(np.iscomplex(z_test)))
    
    # estimator setup (only for RANDOM)
    estimator.setup(z_test[0], z_test[-1])

    # get initial samples
    for j, i in enumerate(start_z_idx):
        z_sample, y, yC = getSample(sampler, z_test, i, j, is_system_selfadjoint)

        if j == 0: # initial sample
            size_y = len(y)
            if not is_system_selfadjoint:
                valsC = np.empty((size_y, len(start_z_idx)), dtype = complex)
        if not is_system_selfadjoint: valsC[:, j : j + 1] = yC

        if j == 0: # initialize rational function
            approx = barycentricFunction(
                        np.empty(len(start_z_idx), dtype = complex),
                        np.empty(len(start_z_idx), dtype = complex),
                        np.empty((size_y, len(start_z_idx)), dtype = complex))
        # update rational function
        approx.supp[j], approx.vals[:, j : j + 1] = z_sample, y
        
        if j == 0: # initialize Loewner matrix
            L = np.empty((size_y * len(start_z_idx), len(start_z_idx)),
                         dtype = complex)
        # update Loewner matrix
        L_new_oldandnew = (1j * (yC - approx.vals[:, : j + 1])
                              / (z_sample + approx.supp[: j + 1]))
        if is_system_selfadjoint:
            L_old_new = L_new_oldandnew[:, :-1].T.conj()
        else:
            L_old_new = 1j * ((valsC - y) / (z_sample + approx.supp[: -1])).T
        L[: j * size_y, j] = L_old_new.flatten()
        L[j * size_y : (j + 1) * size_y, : j + 1] = L_new_oldandnew
    approx.coeffs = getBarycentricCoeffs(L)

    # adaptivity loop
    n_memory = 0
    for j in range(approx.nsupp, Smax): # max number of samples
        # estimator pre-check (only for RANDOM)
        flag = estimator.pre_check(approx)
        if flag == 0: n_memory = 0 # error is too large
        if flag == 1:
            logger.info("termination check passed")
            n_memory += 1 # error is below tolerance

        # termination check
        if n_memory >= N_memory: break # enough small errors in a row

        # find next sample point
        indicator = estimator.indicator(z_test, approx)
        idx_sample = np.argmax(indicator)

        # estimator mid-setup (only for BATCH)
        estimator.mid_setup(z_test, idx_sample, indicator, approx)

        z_sample, y, yC = getSample(sampler, z_test, idx_sample, j, is_system_selfadjoint)

        # estimator post-check (only for LOOK_AHEAD and BATCH)
        flag = estimator.post_check(y, approx)
        if flag == 0: n_memory = 0 # error is too large
        if flag == 1:
            logger.info("termination check passed")
            n_memory += 1 # error is below tolerance
        
        # update surrogate with new support points and values
        approx.supp = np.append(approx.supp, z_sample)
        approx.vals = np.append(approx.vals, y, axis = -1)

        # update Loewner matrix
        L_new_oldandnew = 1j * (yC - approx.vals) / (z_sample + approx.supp)
        if is_system_selfadjoint:
            L_old_new = L_new_oldandnew[:, :-1].T.conj()
        else:
            L_old_new = 1j * ((valsC - y) / (z_sample + approx.supp[: -1])).T
            valsC = np.append(valsC, yC, axis = -1)
        L = np.block([[L, L_old_new.reshape(-1, 1)], [L_new_oldandnew]])
        approx.coeffs = getBarycentricCoeffs(L)

    logger.info("greedy loop terminated at {} samples".format(approx.nsupp))
    if return_estimate:
        return approx, (z_test, estimator.build_eta(z_test, approx))
    return approx
