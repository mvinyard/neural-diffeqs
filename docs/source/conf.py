# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'neural-diffeqs'
copyright = '2023, Michael E. Vinyard'
author = 'Michael E. Vinyard'

# The full version, including alpha/beta/rc tags
release = 'v0.3.2'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_panels',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

html_context = dict(
    github_user="mvinyard",   # Username
    github_repo="neural-diffeqs",   # Repo name
    github_version="main",  # Version
    doc_path="docs/",  # Path in the checkout to the docs root
)

# Set link name generated in the top bar.
html_title = "neural-diffeqs"
html_logo = "../../assets/neural_diffeqs.logo.png"
html_favicon = "../../assets/neural_diffeqs.favicon.png"
html_theme_options = {
    "github_url": "https://github.com/mvinyard/neural-diffeqs",
    "twitter_url": "https://twitter.com/vinyard_m",
}


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output --------------------------------------------------

html_show_sourcelink = True
html_theme = 'pydata_sphinx_theme'

"""
Add paths to css files (e.g., style sheet), below. Such files are copied
after the builtin static files, thus a new file named default.css would
overwrite the builtin default.css.
"""
