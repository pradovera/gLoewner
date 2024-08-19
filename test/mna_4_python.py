# The file mna_4.mat belongs to the SLICOT Benchmark Examples for Model Reduction
# ( Y. Chahlaoui, P. Van Dooren,
#   Benchmark Examples for Model Reduction of Linear Time-Invariant Dynamical Systems, Dimension Reduction of Large-Scale Systems,
#   Lecture Notes in Computational Science and Engineering, vol 45: 379--392, 2005.)
import numpy as np
from scipy.io import loadmat
from scipy.sparse.linalg import spsolve
from gloewner import estimatorLookAhead, trainSurrogate
from matplotlib import pyplot as plt
import logging
logging.basicConfig()
logging.getLogger('gloewner').setLevel(logging.INFO)

problem_data = loadmat("mna_4.mat")
problem_E = problem_data["E"]
problem_A = problem_data["A"]
problem_B = problem_data["B"].toarray()
sampler = lambda z: problem_B.T.conj().dot(
                            spsolve(z * problem_E - problem_A, problem_B))

z_min, z_max = 3e4, 3e9
estimator = estimatorLookAhead(1e-3, 1e-8, sampler)
Smax, N_test, N_memory = 1000, 10000, 1

# train surrogate model
approx, estimate = trainSurrogate(sampler, z_min, z_max, estimator, Smax,
                                  N_test, N_memory, return_estimate = True)

# predict and compute errors
z_post = np.geomspace(z_min, z_max, 101)
H_exact = np.vstack([sampler(1j * z).reshape(1, -1) for z in z_post])
H_approx = approx(z_post)
H_err = estimator.compute_error(H_approx, H_exact) + 1e-16

# make plots
plt.figure()
plt.loglog(z_post, np.abs(H_exact))
plt.loglog(z_post, np.abs(H_approx), '--')
plt.loglog(approx.supp, np.abs(approx.vals).T, 'o')
plt.legend(["H{}".format(i) for i in range(H_exact.shape[1])])
plt.xlabel("Im(z)"), plt.xlabel("|H|")
plt.figure()
plt.loglog(z_post, H_err)
plt.loglog(*estimate, ":")
plt.loglog([z_min, z_max], [estimator.tol] * 2, '--')
plt.legend(["error", "estimator"])
plt.xlabel("Im(z)"), plt.xlabel("relative error")
plt.show()