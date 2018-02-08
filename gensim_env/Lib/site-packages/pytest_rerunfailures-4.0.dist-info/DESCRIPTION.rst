pytest-rerunfailures
====================

pytest-rerunfailures is a plugin for `py.test <http://pytest.org>`_ that
re-runs tests to eliminate intermittent failures.

.. image:: https://img.shields.io/badge/license-MPL%202.0-blue.svg
   :target: https://github.com/pytest-dev/pytest-rerunfailures/blob/master/LICENSE
   :alt: License
.. image:: https://img.shields.io/pypi/v/pytest-rerunfailures.svg
   :target: https://pypi.python.org/pypi/pytest-rerunfailures/
   :alt: PyPI
.. image:: https://img.shields.io/travis/pytest-dev/pytest-rerunfailures.svg
   :target: https://travis-ci.org/pytest-dev/pytest-rerunfailures/
   :alt: Travis

Requirements
------------

You will need the following prerequisites in order to use pytest-rerunfailures:

- Python 2.7, 3.4, 3.5, 3.6, PyPy, or PyPy3
- pytest 2.8.7 or newer

Installation
------------

To install pytest-rerunfailures:

.. code-block:: bash

  $ pip install pytest-rerunfailures

Re-run all failures
-------------------

To re-run all test failures, use the ``--reruns`` command line option with the
maximum number of times you'd like the tests to run:

.. code-block:: bash

  $ pytest --reruns 5

To add a delay time between re-runs use the ``--reruns-delay`` command line
option with the amount of seconds that you would like wait before the next
test re-run is launched:

.. code-block:: bash

   $ pytest --reruns 5 --reruns-delay 1

Re-run individual failures
--------------------------

To mark individual tests as flaky, and have them automatically re-run when they
fail, add the ``flaky`` mark with the maximum number of times you'd like the
test to run:

.. code-block:: python

  @pytest.mark.flaky(reruns=5)
  def test_example():
      import random
      assert random.choice([True, False])

Note that when teardown fails, two reports are generated for the case, one for
the test case and the other for the teardown error.

You can also specify the re-run delay time in the marker:

.. code-block:: python

  @pytest.mark.flaky(reruns=5, reruns_delay=2)
  def test_example():
      import random
      assert random.choice([True, False])

Output
------

Here's an example of the output provided by the plugin when run with
``--reruns 2`` and ``-r aR``::

  test_report.py RRF

  ================================== FAILURES ==================================
  __________________________________ test_fail _________________________________

      def test_fail():
  >       assert False
  E       assert False

  test_report.py:9: AssertionError
  ============================ rerun test summary info =========================
  RERUN test_report.py::test_fail
  RERUN test_report.py::test_fail
  ============================ short test summary info =========================
  FAIL test_report.py::test_fail
  ======================= 1 failed, 2 rerun in 0.02 seconds ====================

Note that output will show all re-runs. Tests that fail on all the re-runs will
be marked as failed.

Compatibility
-------------

* This plugin may *not* be used with class, module, and package level fixtures.
* This plugin is *not* compatible with pytest-xdist's --looponfail flag.
* This plugin is *not* compatible with the core --pdb flag.

Resources
---------

- `Issue Tracker <http://github.com/pytest-dev/pytest-rerunfailures/issues>`_
- `Code <http://github.com/pytest-dev/pytest-rerunfailures/>`_


Changelog
---------


4.0 (2017-12-23)
================

- Added option to add a delay time between test re-runs (Thanks to `@Kanguros`_
  for the PR).

- Added support for pytest >= 3.3.

- Drop support for pytest < 2.8.7.

.. _@Kanguros: https://github.com/Kanguros


3.1 (2017-08-29)
================

- Restored compatibility with pytest-xdist. (Thanks to `@davehunt`_ for the PR)

.. _@davehunt: https://github.com/davehunt


3.0 (2017-08-17)
================

- Add support for Python 3.6.

- Add support for pytest 2.9 up to 3.2

- Drop support for Python 2.6 and 3.3.

- Drop support for pytest < 2.7.


2.2 (2017-06-23)
================

- Ensure that other plugins can run after this one, in case of a global setting
  ``--rerun=0``. (Thanks to `@sublee`_ for the PR)

.. _@sublee: https://github.com/sublee

2.1.0 (2016-11-01)
==================

- Add default value of ``reruns=1`` if ``pytest.mark.flaky()`` is called
  without arguments.

- Also offer a distribution as universal wheel. (Thanks to `@tltx`_ for the PR)

.. _@tltx: https://github.com/tltx


2.0.1 (2016-08-10)
==================

- Prepare CLI options to pytest 3.0, to avoid a deprecation warning.

- Fix error due to missing CHANGES.rst when creating the source distribution
  by adding a MANIFEST.in.


2.0.0 (2016-04-06)
==================

- Drop support for Python 3.2, since supporting it became too much of a hassle.
  (Reason: Virtualenv 14+ / PIP 8+ do not support Python 3.2 anymore.)


1.0.2 (2016-03-29)
==================

- Add support for `--resultlog` option by parsing reruns accordingly. (#28)


1.0.1 (2016-02-02)
==================

- Improve package description and include CHANGELOG into description.


1.0.0 (2016-02-02)
==================

- Rewrite to use newer API of pytest >= 2.3.0

- Improve support for pytest-xdist by only logging the final result.
  (Logging intermediate results will finish the test rather rerunning it.)


