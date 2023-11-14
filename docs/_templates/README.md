# Templates Doc Directory

Add any paths that contain templates here, relative to  
the `conf.py` file's directory.
They are copied after the builtin template files,
so a file named "page.html" will overwrite the builtin "page.html".

The path to this folder is set in the Sphinx `conf.py` file in the line: 
```python
templates_path = ['_templates']
```

The ``custom-class-template.rst`` and ``custom-module-template.rst`` templates are modified versions of the ``autosummary``
default templates to enable the automatic creation of classes and function documentation (see also https://github.com/sphinx-doc/sphinx/issues/7912
and stackoverflow question linked in the issue).
