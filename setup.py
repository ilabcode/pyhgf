import codecs
import os

from setuptools import setup


# Get the package's version number of the __init__.py file
def read(rel_path):
    """Read the file located at the provided relative path."""
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    """Get the package's version number.
    We fetch the version  number from the `__version__` variable located in the
    package root's `__init__.py` file. This way there is only a single source
    of truth for the package's version number.

    """
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


DESCRIPTION = "The generalized, nodalized HGF for predictive coding."
DISTNAME = "ghgf"
AUTHOR = "ILAB"
MAINTAINER = "Nicolas Legrand"
MAINTAINER_EMAIL = "nicolas.legrand@cfin.au.dk"
INSTALL_REQUIRES = [
    "jax>=0.3.13",
    "jaxlib>=0.3.10",
]
PACKAGES = ["ghgf"]



if __name__ == "__main__":

    setup(
        name=DISTNAME,
        author=AUTHOR,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=open("README.md").read(),
        long_description_content_type="text/x-rst",
        license="GPL-3.0",
        version=get_version("ghgf/__init__.py"),
        install_requires=INSTALL_REQUIRES,
        include_package_data=True,
        package_data={"": ["ghgf/ghgf/data/*.dat"]},
        packages=PACKAGES,
    )
