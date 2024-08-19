# gLoewner
Library for greedy Loewner framework. For more details on the algorithm, see

D. Pradovera, _Toward a certified greedy Loewner framework with minimal sampling_, ACOM (2023), DOI: [10.1007/s10444-023-10091-7](https://doi.org/10.1007/s10444-023-10091-7)

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

## License
See the [`LICENSE`](LICENSE) file for licensing information.