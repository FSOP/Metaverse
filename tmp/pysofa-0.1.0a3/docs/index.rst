.. pysofa documentation master file, created by
   sphinx-quickstart on Wed Nov 17 14:57:22 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pysofa's documentation page !
========================================

*pysofa* is a `Python <http://www.python.org/>`_ module for accessing
`International Astronomical Union <http://www.iau.org/>`_'s
`SOFA library <http://www.iausofa.org/>`_ from python. SOFA (Standards
of Fundamental Astronomy) is a set of algorithms and procedures that
implement standard models used in fundamental astronomy.

*pysofa* is not a port of SOFA routines but a wrapper around the SOFA_C
library. Thus, no calculations are made into the pysofa software, they are
all delegated to the underlying SOFA_C library.

Disclaimer
----------

*pysofa* is neither distributed, supported nor endorsed by the International
Astronomical Union. In addition to *pysofa*'s license, any use of this module
should comply with `SOFA's license and terms of use
<http://www.iausofa.org/copyr.pdf>`_. Especially, but not exclusively, any
published work or commercial products which includes results achieved by using
*pysofa* shall acknowledge that the SOFA software was used in obtaining those
results.


Installation
============

Requirements
------------
Before doing anything useful with *pysofa*, you'll need:

    * `Python <http://www.python.org/>`_ 2.5 or higher.
    * `numpy <http://numpy.scipy.org>`_
    * and, obviously, the SOFA_C library.

.. note::
    *pysofa* use `ctypes <http://docs.python.org/library/ctypes.html>`_ to do
    its job, hence, the SOFA_C library must be compiled as a shared library
    and findable by the operating system's dynamic loader. Note that the
    default *makefile* provided with SOFA_C compile the library as a static
    one on UNIX systems.

Install
-------

Once you have the requirements satisfied, you have a few options for
installlation.

If you have `easy_install/setuptools <http://pypi.python.org/pypi/setuptools>`_
or `pip <http://pypi.python.org/pypi/pip>`_ installed, just do::

    pip install pysofa

or::

    easy_install pysofa

If you are installing from source code, do::

    python ./setup.py install


Bug reports
===========
The best place to report bugs or request features is the `google code bug
tracker <http://code.google.com/p/pysofa/issues>`_.


Documentation
=============

.. warning::
    *pysofa* is still alpha software and, although fully functionnal,
    class, method names and calling conventions are subject to change in
    future versions.

Function names of the SOFA library are directly mapped to those of *pysofa*,
with the ``iau`` part stripped. For example, once *pysofa* is loaded into the
interpreter, the original function ``iauDat`` is accessible via
``pysofa.Dat``.

Please refer to the :doc:`reference` page for a complete list of available
functions. The documentation is quite terse, and users are advised to check
the original SOFA manual for detailed explanations of what each function does
and how it works.


.. toctree::
   :maxdepth: 2

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

