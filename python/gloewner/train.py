import numpy as np
from .barycentric import barycentricFunction
from .estimator import samplerEff
from .logging import logger

def trainSurrogate(sampler, z_min, z_max, estimator,
                   Smax, N_test, N_memory = 1,
                   return_estimate = True, is_system_selfadjoint = True):
    """Train rational model using the greedy Loewner framework
    Inputs:
    sampler: Callable to evaluate high-fidelity model. Note: model is
        evaluated at 1j * z.
    z_min, z_max: Smallest and largest values of z.
    estimator: Class of type estimator.estimator to drive the adaptivity.
    Smax: Maximum number of high-fidelity samples to be taken.
    N_test: Number of values of z among which to choose samples.
    N_memory: Depth of memory. Method terminates successfully only if
        estimator yields a "pass" N_memory times in a row.
    return_estimate: Whether to also return an error estimate.
    is_system_selfadjoint: Whether system that generates data (through
        sampler) is selfadjoint:
            sampler(conj(z)) == conj(sampler(z)) for all complex z.
        (In practice, it's enough for the relation to be true on the
        imaginary axis.) If True, saves half the high-fidelity samples.
    """
    z_test = list(np.geomspace(z_min, z_max, N_test))
    # estimator setup (only for RANDOM)
    estimator.setup(z_min, z_max)

    # get initial sample
    z_sample = z_test.pop(0) # remove initial sample point from test set
    y = samplerEff(sampler, z_sample)
    logger.info("0: sampled at z={}j".format(z_sample))
    size = len(y)
    approx = barycentricFunction(np.array([z_sample]), np.ones(1), y)
    if is_system_selfadjoint:
        yC = y.conj()
    else:
        yC = samplerEff(sampler, -z_sample)
        logger.info("0: sampled at z={}j".format(-z_sample))
    L = .5j * (yC - y) / z_sample # Loewner matrix
    if not is_system_selfadjoint: valsC = yC

    # adaptivity loop
    n_memory = 0
    for _ in range(Smax): # max number of samples
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

        z_sample = z_test.pop(idx_sample) # remove sample point from test set
        y = samplerEff(sampler, z_sample) # compute new sample
        logger.info("{}: sampled at z={}j".format(approx.nsupp, z_sample))
        
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
        if is_system_selfadjoint:
            yC = y.conj()
        else:
            yC = samplerEff(sampler, -z_sample)
            logger.info("{}: sampled at z={}j".format(approx.nsupp - 1, -z_sample))
        L1 = 1j * (yC - approx.vals) / (z_sample + approx.supp)
        if is_system_selfadjoint:
            l1 = L1[:, :-1].T.conj()
        else:
            l1 = 1j * ((valsC - y) / (z_sample + approx.supp[: -1])).T
            valsC = np.append(valsC, yC, axis = -1)
        L = np.block([[L, l1.reshape(-1, 1)], [L1]])

        # update surrogate with new barycentric coefficients
        R = np.linalg.qr(L)[1]
        Vh = np.linalg.svd(R)[2]
        approx.coeffs = Vh[-1].conj()

    logger.info("greedy loop terminated at {} samples".format(approx.nsupp))
    if return_estimate:
        return approx, (z_test, estimator.build_eta(z_test, approx))
    return approx
