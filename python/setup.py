import os
from setuptools import find_packages, setup

package_directory = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

setup(name="gLoewner",
      description="Greedy Loewner framework",
      version="1.0.0",
      license="GNU Library or Lesser General Public License (LGPL)",
      packages=find_packages(package_directory),
      zip_safe=False
      )
