import numpy as np

def canonical_j(j:int, n:int):
    # vector of canonical basis
    out = np.zeros(n, dtype = complex)
    out[j] = 1
    return out

class barycentricFunction:
    # class for evaluating rational functions in barycentric form
    def __init__(self, supp, coeffs, vals = None):
        self.supp = supp.flatten() # support points
        self.coeffs = coeffs.flatten() # barycentric (denominator) coefficients
        self.vals = vals # support values (multiplied by coeffs in numerator)

    @property
    def nsupp(self):
        # number of support points
        return len(self.supp)

    @property
    def size(self):
        # size of output
        if self.vals is None: return self.nsupp
        return self.vals.shape[1]

    def __call__(self, x, tol = 1e-10, only_den = False):
        # evaluate function at x (if only_den, evaluate only denominator)
        x = np.array(x).reshape(-1, 1)
        dx = x - self.supp.reshape(1, -1)
        # check if some x is too close to a support point
        # (we compute the samples at such bad x by hand)
        idxs_dx_min = np.argmin(np.abs(dx), axis = 1)
        bad_idxs, bad_idx_vals = [], []
        for j, idx_dx_min in enumerate(idxs_dx_min):
            if np.abs(dx[j, idx_dx_min]) < tol:
                bad_idxs += [j]
                if only_den:
                    pass
                elif self.vals is None:
                    bad_idx_vals += [canonical_j(idx_dx_min, self.nsupp)]
                else:
                    bad_idx_vals += [self.vals[idx_dx_min]]
        good_idxs = list(range(len(x)))
        for j in bad_idxs[::-1]: good_idxs.pop(j)
        # numerator coefficients (to be multiplied by vals)
        num = self.coeffs / dx[good_idxs]
        if only_den:
            den = np.empty(len(x), dtype = complex)
            den[bad_idxs] = np.inf
            den[good_idxs] = np.sum(num, axis = 1)
            return den
        den = np.sum(num, axis = 1)
        if self.vals is None:
            out = np.empty((len(x), self.nsupp), dtype = complex)
        else:
            out = np.empty((len(x), self.size), dtype = complex)
        # compute numerator values
        for j, val in zip(bad_idxs, bad_idx_vals):
            out[j] = val
        if self.vals is None:
            out[good_idxs] = (num.T / den).T
        else:
            out[good_idxs] = (num.dot(self.vals).T / den).T
        return out
