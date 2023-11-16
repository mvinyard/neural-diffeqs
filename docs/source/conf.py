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
    'nbsphinx',
    'sphinx_copybutton',
    'sphinx_favicon',
    'sphinx_design',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output: ------------------------------------------------
html_theme = 'pydata_sphinx_theme'

html_static_path = ['_static']
html_css_files = ['css/custom.css']

html_theme_options = {
    "github_url": "https://github.com/mvinyard/neural-diffeqs",
    "twitter_url": "https://twitter.com/vinyard_m",
    "logo": {
      "image_light": "_static/imgs/neural_diffeqs_logo.simple.light.png",
      "image_dark": "_static/imgs/neural_diffeqs_logo.simple.dark.png",
   },
}
autoclass_content = 'init'



favicons = {"rel": "icon", "href": "imgs/neural_diffeqs.favicon.png"}
