from textwrap import dedent

from functools import wraps
from inspect import getargspec


def generator(*args):
    """
    A utility function for creating a generator that will yield the
    supplied arguments.
    """
    for i in args:
        yield i


class Wrappings:
    def __init__(self):
        self.before = []
        self.after = []


def wrap(before, after=None):
    """
    A decorator that causes the supplied callables to be called before
    or after the wrapped callable, as appropriate.
    """
    def wrapper(wrapped):
        if getattr(wrapped, '_wrappings', None) is None:
            w = Wrappings()

            @wraps(wrapped)
            def wrapping(*args, **kw):
                args = list(args)
                to_add = len(getargspec(wrapped)[0][len(args):])
                added = 0
                for c in w.before:
                    r = c()
                    if added < to_add:
                        args.append(r)
                        added += 1
                try:
                    return wrapped(*args, **kw)
                finally:
                    for c in w.after:
                        c()
            f = wrapping
            f._wrappings = w
        else:
            f = wrapped
        w = f._wrappings
        w.before.append(before)
        if after is not None:
            w.after.insert(0, after)
        return f
    return wrapper


def extend_docstring(docstring, objs):
    for obj in objs:
        try:
            obj.__doc__ = dedent(obj.__doc__) + docstring
        except (AttributeError, TypeError):  # python 2 or pypy 4.0.1 :-(
            pass
