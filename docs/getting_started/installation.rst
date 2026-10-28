Installation
============

Requirements
------------

PyQCA requires Python 3.11 or newer. NumPy and pandas are installed with the
base package.

Release installation
--------------------

This documentation prepares version 0.3.0. Once it is published to PyPI:

.. code-block:: console

   python -m pip install "pyqca==0.3.0"
   python -m pip install "pyqca[viz,mlqca]==0.3.0"

Before publication, use the source installation below. The distribution name
is ``pyqca``; the Python import name is ``qca``. Python 3.11 and 3.12 are
tested in CI. See :doc:`../release_notes` before upgrading an existing analysis.

Development install
-------------------

Until the first PyPI release, install directly from a local clone:

.. code-block:: console

   git clone https://github.com/t-yamsaki/PyQCA.git
   cd PyQCA
   python -m pip install -e .

Optional features
-----------------

Install only the feature groups needed by the analysis:

.. code-block:: console

   # Matplotlib-based plots
   python -m pip install -e ".[viz]"

   # XGBoost and scikit-learn for qca.mlqca
   python -m pip install -e ".[mlqca]"

   # mlQCA plus the standalone SHAP package
   python -m pip install -e ".[mlqca-explain]"

   # Build this documentation locally
   python -m pip install -e ".[docs]"

The base QCA engines do not require XGBoost, scikit-learn, SHAP, or
Matplotlib.

Verify the installation
-----------------------

.. code-block:: python

   import qca

   print(qca.__version__)
   print(qca.available_minimizers())
