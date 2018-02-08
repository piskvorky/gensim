.. image:: http://www.attrs.org/en/latest/_static/attrs_logo.png
   :alt: attrs Logo

======================================
``attrs``: Classes Without Boilerplate
======================================

.. image:: https://readthedocs.org/projects/attrs/badge/?version=stable
   :target: http://www.attrs.org/en/stable/?badge=stable
   :alt: Documentation Status

.. image:: https://travis-ci.org/python-attrs/attrs.svg?branch=master
   :target: https://travis-ci.org/python-attrs/attrs
   :alt: CI Status

.. image:: https://codecov.io/github/python-attrs/attrs/branch/master/graph/badge.svg
   :target: https://codecov.io/github/python-attrs/attrs
   :alt: Test Coverage

.. teaser-begin

``attrs`` is the Python package that will bring back the **joy** of **writing classes** by relieving you from the drudgery of implementing object protocols (aka `dunder <https://nedbatchelder.com/blog/200605/dunder.html>`_ methods).

Its main goal is to help you to write **concise** and **correct** software without slowing down your code.

.. -spiel-end-

For that, it gives you a class decorator and a way to declaratively define the attributes on that class:

.. -code-begin-

.. code-block:: pycon

   >>> import attr

   >>> @attr.s
   ... class SomeClass(object):
   ...     a_number = attr.ib(default=42)
   ...     list_of_numbers = attr.ib(default=attr.Factory(list))
   ...
   ...     def hard_math(self, another_number):
   ...         return self.a_number + sum(self.list_of_numbers) * another_number


   >>> sc = SomeClass(1, [1, 2, 3])
   >>> sc
   SomeClass(a_number=1, list_of_numbers=[1, 2, 3])

   >>> sc.hard_math(3)
   19
   >>> sc == SomeClass(1, [1, 2, 3])
   True
   >>> sc != SomeClass(2, [3, 2, 1])
   True

   >>> attr.asdict(sc)
   {'a_number': 1, 'list_of_numbers': [1, 2, 3]}

   >>> SomeClass()
   SomeClass(a_number=42, list_of_numbers=[])

   >>> C = attr.make_class("C", ["a", "b"])
   >>> C("foo", "bar")
   C(a='foo', b='bar')


After *declaring* your attributes ``attrs`` gives you:

- a concise and explicit overview of the class's attributes,
- a nice human-readable ``__repr__``,
- a complete set of comparison methods,
- an initializer,
- and much more,

*without* writing dull boilerplate code again and again and *without* runtime performance penalties.

This gives you the power to use actual classes with actual types in your code instead of confusing ``tuple``\ s or `confusingly behaving <http://www.attrs.org/en/stable/why.html#namedtuples>`_ ``namedtuple``\ s.
Which in turn encourages you to write *small classes* that do `one thing well <https://www.destroyallsoftware.com/talks/boundaries>`_.
Never again violate the `single responsibility principle <https://en.wikipedia.org/wiki/Single_responsibility_principle>`_ just because implementing ``__init__`` et al is a painful drag.


.. -testimonials-

Testimonials
============

**Amber Hawkie Brown**, Twisted Release Manager and Computer Owl:

  Writing a fully-functional class using attrs takes me less time than writing this testimonial.


**Glyph Lefkowitz**, creator of `Twisted <https://twistedmatrix.com/>`_, `Automat <https://pypi.python.org/pypi/Automat>`_, and other open source software, in `The One Python Library Everyone Needs <https://glyph.twistedmatrix.com/2016/08/attrs.html>`_:

  I’m looking forward to is being able to program in Python-with-attrs everywhere.
  It exerts a subtle, but positive, design influence in all the codebases I’ve see it used in.


**Kenneth Reitz**, author of `requests <http://www.python-requests.org/>`_, Python Overlord at Heroku, `on paper no less <https://twitter.com/hynek/status/866817877650751488>`_:

  attrs—classes for humans.  I like it.


**Łukasz Langa**, prolific CPython core developer and Production Engineer at Facebook:

  I'm increasingly digging your attr.ocity. Good job!


.. -end-

.. -project-information-

Getting Help
============

Please use the ``python-attrs`` tag on `StackOverflow <https://stackoverflow.com/questions/tagged/python-attrs>`_ to get help.

Answering questions of your fellow developers is also great way to help the project!


Project Information
===================

``attrs`` is released under the `MIT <https://choosealicense.com/licenses/mit/>`_ license,
its documentation lives at `Read the Docs <http://www.attrs.org/>`_,
the code on `GitHub <https://github.com/python-attrs/attrs>`_,
and the latest release on `PyPI <https://pypi.org/project/attrs/>`_.
It’s rigorously tested on Python 2.7, 3.4+, and PyPy.

If you'd like to contribute you're most welcome and we've written `a little guide <http://www.attrs.org/en/latest/contributing.html>`_ to get you started!


Release Information
===================

17.4.0 (2017-12-30)
-------------------

Backward-incompatible Changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- The traversal of MROs when using multiple inheritance was backward:
  If you defined a class ``C`` that subclasses ``A`` and ``B`` like ``C(A, B)``, ``attrs`` would have collected the attributes from ``B`` *before* those of ``A``.

  This is now fixed and means that in classes that employ multiple inheritance, the output of ``__repr__`` and the order of positional arguments in ``__init__`` changes.
  Due to the nature of this bug, a proper deprecation cycle was unfortunately impossible.

  Generally speaking, it's advisable to prefer ``kwargs``-based initialization anyways – *especially* if you employ multiple inheritance and diamond-shaped hierarchies.

  `#298 <https://github.com/python-attrs/attrs/issues/298>`_,
  `#299 <https://github.com/python-attrs/attrs/issues/299>`_,
  `#304 <https://github.com/python-attrs/attrs/issues/304>`_
- The ``__repr__`` set by ``attrs``
  no longer produces an ``AttributeError``
  when the instance is missing some of the specified attributes
  (either through deleting
  or after using ``init=False`` on some attributes).

  This can break code
  that relied on ``repr(attr_cls_instance)`` raising ``AttributeError``
  to check if any attr-specified members were unset.

  If you were using this,
  you can implement a custom method for checking this::

      def has_unset_members(self):
          for field in attr.fields(type(self)):
              try:
                  getattr(self, field.name)
              except AttributeError:
                  return True
          return False

  `#308 <https://github.com/python-attrs/attrs/issues/308>`_


Deprecations
^^^^^^^^^^^^

- The ``attr.ib(convert=callable)`` option is now deprecated in favor of ``attr.ib(converter=callable)``.

  This is done to achieve consistency with other noun-based arguments like *validator*.

  *convert* will keep working until at least January 2019 while raising a ``DeprecationWarning``.

  `#307 <https://github.com/python-attrs/attrs/issues/307>`_


Changes
^^^^^^^

- Generated ``__hash__`` methods now hash the class type along with the attribute values.
  Until now the hashes of two classes with the same values were identical which was a bug.

  The generated method is also *much* faster now.

  `#261 <https://github.com/python-attrs/attrs/issues/261>`_,
  `#295 <https://github.com/python-attrs/attrs/issues/295>`_,
  `#296 <https://github.com/python-attrs/attrs/issues/296>`_
- ``attr.ib``\ ’s ``metadata`` argument now defaults to a unique empty ``dict`` instance instead of sharing a common empty ``dict`` for all.
  The singleton empty ``dict`` is still enforced.

  `#280 <https://github.com/python-attrs/attrs/issues/280>`_
- ``ctypes`` is optional now however if it's missing, a bare ``super()`` will not work in slots classes.
  This should only happen in special environments like Google App Engine.

  `#284 <https://github.com/python-attrs/attrs/issues/284>`_,
  `#286 <https://github.com/python-attrs/attrs/issues/286>`_
- The attribute redefinition feature introduced in 17.3.0 now takes into account if an attribute is redefined via multiple inheritance.
  In that case, the definition that is closer to the base of the class hierarchy wins.

  `#285 <https://github.com/python-attrs/attrs/issues/285>`_,
  `#287 <https://github.com/python-attrs/attrs/issues/287>`_
- Subclasses of ``auto_attribs=True`` can be empty now.

  `#291 <https://github.com/python-attrs/attrs/issues/291>`_,
  `#292 <https://github.com/python-attrs/attrs/issues/292>`_
- Equality tests are *much* faster now.

  `#306 <https://github.com/python-attrs/attrs/issues/306>`_
- All generated methods now have correct ``__module__``, ``__name__``, and (on Python 3) ``__qualname__`` attributes.

  `#309 <https://github.com/python-attrs/attrs/issues/309>`_

`Full changelog <http://www.attrs.org/en/stable/changelog.html>`_.

Credits
=======

``attrs`` is written and maintained by `Hynek Schlawack <https://hynek.me/>`_.

The development is kindly supported by `Variomedia AG <https://www.variomedia.de/>`_.

A full list of contributors can be found in `GitHub's overview <https://github.com/python-attrs/attrs/graphs/contributors>`_.

It’s the spiritual successor of `characteristic <https://characteristic.readthedocs.io/>`_ and aspires to fix some of it clunkiness and unfortunate decisions.
Both were inspired by Twisted’s `FancyEqMixin <https://twistedmatrix.com/documents/current/api/twisted.python.util.FancyEqMixin.html>`_ but both are implemented using class decorators because `sub-classing is bad for you <https://www.youtube.com/watch?v=3MNVP9-hglc>`_, m’kay?


