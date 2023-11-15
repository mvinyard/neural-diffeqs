__doc__ = """Configuration file for the Sphinx documentation builder."""

# -- Project information: -----------------------------------------------------
project = 'neural-diffeqs'
copyright = '2023, Michael E. Vinyard'
author = 'Michael E. Vinyard'
release = '0.3.2'

# -- General configuration: ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output: ------------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_theme_options = {
    "github_url": "https://github.com/mvinyard/neural-diffeqs",
    "twitter_url": "https://twitter.com/vinyard_m",
    "logo": {
      "image_light": "_static/neural_diffeqs_logo.simple.light.png",
      "image_dark": "_static/neural_diffeqs_logo.simple.dark.png",
   }
}

html_logo = "_static/neural_diffeqs_logo.simple.dark.png"
html_css_files = ['_static/custom.css']
