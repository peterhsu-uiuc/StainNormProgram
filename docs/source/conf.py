# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'StainNormProgram'
copyright = '2026, peterhsu-uiuc'
author = 'peterhsu-uiuc'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',
    'autoapi.extension',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']

# -- AutoAPI -----------------------------------------------------------------
autoapi_dirs = ["../../src/stainnormprogram"]
autoapi_python_class_content = "both"
autodoc_typehints = "both"

def skip_util_classes(app, what, name, obj, skip, options):
    if what == "attribute":
       skip = True
    return skip

def setup(sphinx):
   sphinx.connect("autoapi-skip-member", skip_util_classes)

# -- Theme Settings -----------------------------------------------------------
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/peterhsu-uiuc/StainNormProgram",
            "icon": "fa-brands fa-github",
        }
    ],
    "switcher": {
        "json_url": "https://stainnormprogram.readthedocs.io/en/latest/_static/switcher.json",
        "version_match": "1.0.0",
    },
    "navbar_start": ["navbar-logo", "version-switcher"],
}