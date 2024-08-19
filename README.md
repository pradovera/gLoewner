# gLoewner
Library for greedy Loewner framework. For more details on the algorithm, see

> D. Pradovera, _Toward a certified greedy Loewner framework with minimal sampling_, ACOM (2023), DOI: [10.1007/s10444-023-10091-7](https://doi.org/10.1007/s10444-023-10091-7)

With `gLoewner`, one can approximate arbitrary functions using rational approximants in barycentric form, given a collection of samples. The algorithm automatically selects the best locations for new samples using a greedy strategy.

For convenience, the library assumes that one wishes to approximate a univariate target function `f(z)` using samples on the imaginary axis, i.e., given _real_ bounds `0<=zmin<zmax`, the method is allowed to take samples at any position of the form `1j*z` and `-1j*z`, with `zmin<=z<=zmax`. Instead, approximating `f` using _real_ samples is possible by artificially multiplying the interval bounds `zmin` and `zmax` by `-1j`.

Note that, if the function `f` is self-adjoint, sampling the negative imaginary axis is easier: `f(-1j*z)=conj(f(1j*z))`. The algorithm takes advantage of this to save computational resources.

## Python install
Prerequisites:
* **python** (version>=3.8.12)
* **numpy** (version>=1.21.4)
* **scipy** (version>=1.5.3)
* **matplotlib** (version>=3.4.3) for testing

From inside the [`python`](python) folder, you can install the package from a terminal by typing
```
pip install .
```

Then you can import the library from Python as
```
import gloewner
```

Note that the Python library logs information via the `logging` library. To display the logs, use commands like
```
import logging
logging.basicConfig()
logging.getLogger('gloewner').setLevel(logging.INFO)
```
See also the test scripts named `*_python.py` in the [`test`](test) folder.

## MATLAB:tm: install
Prerequisites:
* **matlab** (version>=R2022b)

You need to add the [`matlab`](matlab) folder to the MATLAB path. From inside the folder, you can do this in MATLAB by typing
```
addpath(gLoewnerRoot), savepath
```

## License
See the [`LICENSE`](LICENSE) file for licensing information.
