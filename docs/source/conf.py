# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import time

import pyhgf

# -- Project information -----------------------------------------------------

project = "pyhgf"
copyright = u"2022-{}, Nicolas Legrand".format(time.strftime("%Y"))
author = "Nicolas Legrand"
release = pyhgf.__version__

nb_execution_timeout = 300

image_scrapers = ("matplotlib",)

bibtex_bibfiles = ['refs.bib']
bibtex_reference_style = "author_year"
bibtex_default_style = "unsrt"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "matplotlib.sphinxext.plot_directive",
    "numpydoc",
    "myst_nb",
    "sphinxcontrib.bibtex",
    "sphinx_togglebutton",
    "sphinx_design"
]

panels_add_bootstrap_css = False

# Generate the API documentation when building
autosummary_generate = True
numpydoc_show_class_members = False

# raise an error if the documentation does not build and exit the process
# this should especially ensure that the notebooks run correctly
nb_execution_raise_on_error = True

# Include the example source for plots in API docs
plot_include_source = True
plot_formats = [("png", 90)]
plot_html_show_formats = False
plot_html_show_source_link = False

source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["*.ipynb"]

# -- Options for HTML output -------------------------------------------------

html_logo = "images/logo_small.svg"
html_favicon = "images/logo_small.svg"

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "icon_links": [
        dict(
            name="GitHub",
            url="https://github.com/ilabcode/pyhgf",
            icon="fa-brands fa-square-github",
        ),
        dict(
            name="Twitter",
            url="https://mastodon.social/@nicolegrand",
            icon="fa-brands fa-mastodon",
        ),
        dict(
            name="Pypi",
            url="https://pypi.org/project/pyhgf/",
            icon="fa-solid fa-box",
        ),
    ],
    "logo": {
        "text": "pyhgf",
    },
}

myst_enable_extensions = ["dollarmath", "colon_fence"]

html_sidebars = {
  "api": [],
}
