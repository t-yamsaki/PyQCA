Installation
============

Requirements
------------

PyQCA requires Python 3.11 or newer. NumPy and pandas are installed with the
base package.

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
