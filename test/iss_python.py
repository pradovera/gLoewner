# The file iss.mat belongs to the SLICOT Benchmark Examples for Model Reduction
# ( Y. Chahlaoui, P. Van Dooren,
#   Benchmark Examples for Model Reduction of Linear Time-Invariant Dynamical Systems, Dimension Reduction of Large-Scale Systems,
#   Lecture Notes in Computational Science and Engineering, vol 45: 379--392, 2005.)
import numpy as np
from scipy.io import loadmat
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve
from gloewner import (estimatorLookAhead, estimatorLookAheadBatch,
                      estimatorRandom, trainSurrogate)
from matplotlib import pyplot as plt
import logging
logging.basicConfig()
logging.getLogger('gloewner').setLevel(logging.INFO)

problem_data = loadmat("iss.mat")
problem_A = problem_data["A"]
problem_E = eye(problem_A.shape[0], format = problem_A.format)
problem_B = problem_data["B"].toarray()
problem_C = problem_data["C"].toarray()
sampler = lambda z: problem_C.dot(
                            spsolve(z * problem_E - problem_A, problem_B))

z_min, z_max = 1e-1, 5e1
estimators = [estimatorLookAhead(1e-3, 1e-8, sampler),
              estimatorLookAheadBatch(1e-3, 1e-8, sampler, 5),
              estimatorRandom(1e-3, 1e-8, sampler, 100, 42)]
Smax, N_test, N_memory = 1000, 10000, 3

for estimator in estimators:
    # train surrogate model
    approx = trainSurrogate(sampler, z_min, z_max, estimator, Smax, N_test, N_memory)

    # predict and compute errors
    z_post = np.geomspace(z_min, z_max, 101)
    H_exact = np.vstack([sampler(1j * z).reshape(1, -1) for z in z_post])
    H_approx = approx(z_post)
    H_err = estimator.compute_error(H_approx, H_exact) + 1e-16

    # update test set (for plotting the estimator only)
    z_test = np.geomspace(z_min, z_max, N_test)
    for z in approx.supp:
        idx_bad = np.argmin(np.abs(z_test - z))
        z_test = np.append(z_test[: idx_bad], z_test[idx_bad + 1 :])
    eta = estimator.build_eta(z_test, approx)

    # make plots
    plt.figure()
    plt.loglog(z_post, np.abs(H_exact))
    plt.loglog(z_post, np.abs(H_approx), '--')
    plt.loglog(approx.supp, np.abs(approx.vals), 'o')
    plt.legend(["H{}".format(i) for i in range(H_exact.shape[1])])
    plt.xlabel("Im(z)"), plt.xlabel("|H|")
    plt.title(type(estimator).__name__)
    plt.figure()
    plt.loglog(z_post, H_err)
    plt.loglog(z_test, eta, ":")
    plt.loglog([z_min, z_max], [estimator.tol] * 2, '--')
    plt.legend(["error", "estimator"])
    plt.xlabel("Im(z)"), plt.xlabel("relative error")
    plt.title(type(estimator).__name__)
    plt.show()