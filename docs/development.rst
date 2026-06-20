Development and documentation
=============================

Contributing
------------

Contributions are accepted under the project's MIT License. Read
``CONTRIBUTING.md`` before opening a pull request. Changes involving adapted
methods, external datasets, or reproduction fixtures must document source
provenance and license compatibility.

Local documentation build
-------------------------

.. code-block:: console

   python -m pip install -e ".[docs]"
   sphinx-build -W --keep-going -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` after the build completes.

Read the Docs
-------------

The repository-level ``.readthedocs.yaml`` selects a fixed build image and
Python version, installs the package with its ``docs`` extra, and builds
``docs/conf.py``. Import the GitHub repository in Read the Docs; no manual
build command is required.

Pull request previews should be enabled in the Read the Docs project settings
so documentation changes can be reviewed before merging.

GitHub Actions
--------------

Pull requests targeting ``main`` run:

* the complete test suite on Python 3.11 and 3.12;
* Ruff, isort, and flake8;
* source and wheel builds followed by Twine metadata validation;
* Sphinx with warnings treated as errors.

Configure the ``main`` branch ruleset to require ``Tests (Python 3.11)``,
``Tests (Python 3.12)``, ``Lint and package``, and ``Sphinx`` before merging.
The workflows only report quality status; they do not merge pull requests,
create tags, or publish releases.

Documentation conventions
-------------------------

* User-facing documentation and public docstrings use English.
* Keep examples executable against the current public API.
* Add new public objects to the appropriate API reference page.
* Run Sphinx with ``-W`` so broken references and warnings fail the build.
* Keep the Japanese README synchronized for major user-facing features.
