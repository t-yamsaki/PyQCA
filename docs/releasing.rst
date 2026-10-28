Preparing and publishing a release
==================================

Version and verification
------------------------

Keep pyproject.toml, src/qca/_version.py and CITATION.cff versions equal.
Sphinx reads the runtime version. Update :doc:`release_notes` with behavior
changes and migration notes. Add date-released to CITATION.cff only when
the actual release date is known; do not carry forward an old release date.

From a clean checkout with the proposed release changes:

.. code-block:: console

   python -m pip install -e ".[dev,mlqca]"
   python -m pytest
   python -m sphinx -W --keep-going -b html docs docs/_build/html
   python -m build
   python -m twine check --strict dist/*

Use a fresh output directory so old versions cannot be uploaded accidentally.
Install the wheel in a separate environment and check that
qca.__version__ equals importlib.metadata.version("pyqca").
Confirm the sdist also builds a wheel, the README renders, and license files
are present. Review artifact contents and author metadata before publication.

Trusted Publishing setup
------------------------

The repository already contains separate PyPI and TestPyPI workflows.
They use OIDC Trusted Publishing, not a stored API token. Configure each
service separately with these values:

.. list-table::
   :header-rows: 1

   * - Setting
     - PyPI
     - TestPyPI
   * - Project
     - pyqca
     - pyqca
   * - GitHub owner
     - t-yamsaki
     - t-yamsaki
   * - Repository
     - PyQCA
     - PyQCA
   * - Workflow filename
     - publish.yml
     - publish-testpypi.yml
   * - GitHub environment
     - pypi
     - testpypi

For a first upload, configure a pending publisher in the service account;
for an existing project, configure its trusted publisher. A pending publisher
does not reserve the project name. See the
`official PyPI guide <https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/>`_.

Create the matching GitHub environments and choose protection rules suited
to the maintainer team. A required reviewer must be someone who can actually
approve the deployment. These account and environment settings are external
to the repository and must be verified before publishing.

Release sequence
----------------

1. Merge the release preparation PR only after all CI checks and review pass.
2. Run Publish to TestPyPI manually with main selected. Other branches are
   skipped by the workflow. Check the uploaded description and install the
   package in a fresh environment.
3. Install dependencies from PyPI, then the candidate from TestPyPI without
   dependency resolution, avoiding reliance on TestPyPI's dependency inventory:

   .. code-block:: console

      python -m pip install "numpy>=1.26" "pandas>=2.1"
      python -m pip install --index-url https://test.pypi.org/simple/ --no-deps "pyqca==0.3.0"
      python -c "import qca; print(qca.__version__)"

4. Prepare the release citation date and final release notes in the reviewed
   commit. Create tag v0.3.0 on that commit and publish the GitHub Release.
   Publishing the Release triggers publish.yml; pushing a tag alone does not.
5. Confirm the PyPI upload and install pyqca==0.3.0 in a fresh environment.
   Enable/build the release tag on Read the Docs and verify the versioned docs.

The PyPI workflow checks that the tag matches the package version.
Uploaded distribution filenames cannot be reused for changed contents.
If a TestPyPI candidate needs changes, choose a new version for its next upload.
Do not publish a GitHub Release until the upload is intended.
