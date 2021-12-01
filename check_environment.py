"""
Check Environment
"""
from sys import executable, version as sysv
from jupyter import __file__
import pip
import numpy
import matplotlib
import sklearn
import scipy
import pandas
import PIL
import seaborn
import tensorflow


def check_version(pkg, version):
    """
    check version function
    """
    actual = pkg.__version__.split(".")
    if len(actual) >= 2:
        actual_major = ".".join(actual)
    else:
        raise NotImplementedError(pkg.__name__ + "actual version :" + pkg.__version__)
    try:
        assert actual_major == version
    except Exception as ex:
        print("{} {}\t=> {}".format(pkg.__name__, version, pkg.__version__))
        raise ex


if __name__ == "__main__":
    print(executable)
    print(sysv)
    print(__file__)
    check_version(pip, "21.3.1")
    check_version(numpy, "1.19.5")
    check_version(matplotlib, "3.5.0")
    check_version(sklearn, "1.0.1")
    check_version(scipy, "1.7.3")
    check_version(pandas, "1.3.4")
    check_version(PIL, "8.4.0")
    check_version(seaborn, "0.11.2")
    check_version(tensorflow, "2.6.0")
    print("Houston we are go!")
