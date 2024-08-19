import numpy as np
from .barycentric import barycentricFunction
from .estimator import samplerEff
from .logging import logger

# size = engine.size
def trainSurrogate(sampler, z_min, z_max, estimator,
                   Smax, N_test, N_memory = 1):
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
    """
    z_test = list(np.geomspace(z_min, z_max, N_test))
    # estimator setup (only for RANDOM)
    estimator.setup(z_min, z_max)

    # get initial sample
    z_sample = z_test.pop(0) # remove initial sample point from test set
    y = samplerEff(sampler, z_sample)
    logger.info("0: sampled at z={}j".format(z_sample))
    size = len(y)
    approx = barycentricFunction(np.array([z_sample]), np.ones(1),
                                 np.expand_dims(y, 0))
    L = np.expand_dims(.5j * (y.conj() - y) / z_sample, -1) # Loewner matrix

    # adaptivity loop
    n_memory = 0
    for _ in range(Smax): # max number of samples
        # estimator pre-check (only for RANDOM)
        flag = estimator.pre_check(approx)
        if flag == 0: n_memory = 0 # error is too large
        if flag == 1: n_memory += 1 # error is below tolerance

        # termination check
        if n_memory >= N_memory: break # enough small errors in a row

        # find next sample point
        indicator = estimator.indicator(z_test, approx)
        idx_sample = np.argmax(indicator)

        # estimator mid-setup (only for BATCH)
        estimator.mid_setup(z_test, idx_sample, indicator, approx)

        z_sample = z_test.pop(idx_sample) # remove sample point from test set
        y = samplerEff(sampler, z_sample) # compute new sample
        
        # estimator post-check (only for LOOK_AHEAD and BATCH)
        flag = estimator.post_check(y, approx)
        if flag == 0: n_memory = 0 # error is too large
        if flag == 1: n_memory += 1 # error is below tolerance
        logger.info("{}: sampled at z={}j".format(approx.nsupp, z_sample))
        
        # update surrogate with new support points and values
        approx.supp = np.append(approx.supp, z_sample)
        approx.vals = np.append(approx.vals, np.expand_dims(y, 0), axis = 0)

        # update Loewner matrix
        L = np.pad(L, [(0, size), (0, 1)])
        for j, (k, s) in enumerate(zip(approx.supp, approx.vals)):
            if j < approx.nsupp - 1: # right column
                L[j * size : (j + 1) * size, -1] = (1j * (s.conj() - y)
                                                       / (k + z_sample))
        L[- size :, :] = (1j * (approx.vals[[-1]].conj() - approx.vals).T
                                                / (z_sample + approx.supp))

        # update surrogate with new barycentric coefficients
        approx.coeffs = np.linalg.svd(np.linalg.qr(L)[1])[2][-1].conj()

    logger.info("greedy loop terminated at {} samples".format(approx.nsupp))
    return approx