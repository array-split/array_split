===========================
Contributing to array_split
===========================

.. Start of sphinx doc include.

Welcome! We appreciate your interest in contributing to ``array_split``. 
If you haven't done so already, check out the
`README <https://github.com/array-split/array_split/blob/dev/README.rst>`_

How to contribute
=================

Workflow
--------

The preferred workflow for contributing to ``array_split`` is to fork the
`array_split repository <https://github.com/array-split/array_split>`_ on
GitHub, clone, and develop on a branch. Steps:

1. Fork the `array_split repository <https://github.com/array-split/array_split>`_
   by clicking on the 'Fork' button near the top right of the page. This creates
   a copy of the code under your GitHub user account. For more details on
   how to fork a repository see `this guide <https://help.github.com/articles/fork-a-repo/>`_.

2. Clone your fork of the ``array_split`` repo from your GitHub account to your local disk::

   $ git clone git@github.com:YourLogin/array_split.git
   $ cd array_split

3. Create a ``feature`` branch to hold your development changes::

   $ git checkout -b my-feature

   Always use a ``feature`` branch. It's good practice to never work on the ``master`` branch!

4. Develop the feature on your feature branch. Add changed files
   using ``git add`` and then ``git commit`` files::

      $ git add modified_files
      $ git commit

   to record your changes in Git, then push the changes to your GitHub account with::

      $ git push -u origin my-feature

5. Follow 
   `these instructions <https://help.github.com/articles/creating-a-pull-request-from-a-fork>`_
   to create a pull request from your fork. This will send an email to the committers.

(If any of the above seems like magic to you, please look up the
`Git documentation <https://git-scm.com/documentation>`_ online.

Coding Guidelines
-----------------

1. Unit test new code using python `unittest <https://docs.python.org/3/library/unittest.html>`_
   framework.

2. Ensure `unittest <https://docs.python.org/3/library/unittest.html>`_ coverage is good (``>90%``)
   by using the `coverage <https://pypi.python.org/pypi/coverage>`_ tool::
   
      $ coverage run --source=array_split --omit='*logging*,*unittest*,*rtd*' -m array_split.tests
      $ coverage report -m

3. Ensure style by using `autopep8 <https://pypi.python.org/pypi/autopep8>`_
   and `flake8 <https://pypi.python.org/pypi/flake8>`_ compliance::

      $ autopep8 -r -i -a --max-line-length=100 array_split
      $ flake8 array_split

4. Use docstrings for API documentation and ensure that it builds with sphinx (without warnings)
   and renders correctly::
   
      $ python setup.py build_sphinx

   produces top level html file ``docs/_build/html/index.html``.

Code of Conduct
---------------

``array_split`` adheres to the
`Python Code Quality Authority’s Code of Conduct <http://meta.pycqa.org/en/latest/code-of-conduct.html>`_.
 
