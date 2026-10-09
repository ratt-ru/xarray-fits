.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/ska-sa/xarray-fits/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

xarray-fits could always use more documentation, whether as part of the
official xarray-fits docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/ska-sa/xarray-fits/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `xarray-fits` for local development.

1. Fork the `xarray-fits` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/xarray-fits.git

3. Install your local copy and its development tools with uv::

    $ cd xarray-fits/
    $ uv sync --all-groups
    $ uv run pre-commit install

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass the
   linters, type checks and tests::

    $ uv run pre-commit run -a
    $ uv run pytest -s -vvv tests/ -Werror

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 3.11 and later. Check that
   the GitHub Actions tests pass for all supported Python versions.

Tips
----

To run a subset of tests::

$ uv run pytest tests/test_image_dataset.py


Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in docs/source/changelog.rst).
Then run::

$ uv run tbump --dry-run X.Y.Z
$ uv run tbump X.Y.Z

GitHub Actions will then deploy to PyPI if tests pass.
