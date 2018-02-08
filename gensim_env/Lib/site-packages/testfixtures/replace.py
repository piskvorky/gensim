from functools import partial
from testfixtures.compat import ClassType
from testfixtures.resolve import resolve, not_there
from testfixtures.utils import wrap, extend_docstring

import warnings


def not_same_descriptor(x, y, descriptor):
    return isinstance(x, descriptor) and not isinstance(y, descriptor)


class Replacer:
    """
    These are used to manage the mocking out of objects so that units
    of code can be tested without having to rely on their normal
    dependencies.
    """

    def __init__(self):
        self.originals = {}

    def _replace(self, container, name, method, value, strict=True):
        if value is not_there:
            if method == 'a':
                delattr(container, name)
            if method == 'i':
                del container[name]
        else:
            if method == 'a':
                setattr(container, name, value)
            if method == 'i':
                container[name] = value

    def __call__(self, target, replacement, strict=True):
        """
        Replace the specified target with the supplied replacement.
        """

        container, method, attribute, t_obj = resolve(target)
        if method is None:
            raise ValueError('target must contain at least one dot!')
        if t_obj is not_there and strict:
            raise AttributeError('Original %r not found' % attribute)
        if t_obj is not_there and replacement is not_there:
            return not_there

        replacement_to_use = replacement

        if isinstance(container, (type, ClassType)):

            if not_same_descriptor(t_obj, replacement, classmethod):
                replacement_to_use = classmethod(replacement)

            elif not_same_descriptor(t_obj, replacement, staticmethod):
                replacement_to_use = staticmethod(replacement)

        self._replace(container, attribute, method, replacement_to_use, strict)
        if target not in self.originals:
            self.originals[target] = t_obj
        return replacement

    def replace(self, target, replacement, strict=True):
        """
        Replace the specified target with the supplied replacement.
        """
        self(target, replacement, strict)

    def restore(self):
        """
        Restore all the original objects that have been replaced by
        calls to the :meth:`replace` method of this :class:`Replacer`.
        """
        for target, original in tuple(self.originals.items()):
            container, method, attribute, found = resolve(target)
            self._replace(container, attribute, method, original, strict=False)
            del self.originals[target]

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.restore()

    def __del__(self):
        if self.originals:
            # no idea why coverage misses the following statement
            # it's covered by test_replace.TestReplace.test_replacer_del
            warnings.warn(  # pragma: no cover
                'Replacer deleted without being restored, '
                'originals left: %r' % self.originals
                )


def replace(target, replacement, strict=True):
    """
    A decorator to replace a target object for the duration of a test
    function.
    """
    r = Replacer()
    return wrap(partial(r.__call__, target, replacement, strict), r.restore)


class Replace(object):
    """
    A context manager that uses a :class:`Replacer` to replace a single target.
    """

    def __init__(self, target, replacement, strict=True):
        self.target = target
        self.replacement = replacement
        self.strict = strict
        self._replacer = Replacer()

    def __enter__(self):
        return self._replacer(self.target, self.replacement, self.strict)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._replacer.restore()

replace_params_doc = """
:param target: A string containing the dotted-path to the
               object to be replaced. This path may specify a
               module in a package, an attribute of a module,
               or any attribute of something contained within
               a module.

:param replacement: The object to use as a replacement.

:param strict: When `True`, an exception will be raised if an
               attempt is made to replace an object that does
               not exist.
"""

# add the param docs, so we only have one copy of them!
extend_docstring(replace_params_doc,
                 [Replacer.__call__, Replacer.replace, replace, Replace])
