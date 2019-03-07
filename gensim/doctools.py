#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors: Michael Penkov <m@penkov.id.au>
# Copyright (C) 2019 RaRe Technologies s.r.o.
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
"""Common functions for working with docstrings.

For internal use only.
"""
import inspect
import io


def extract_kwargs(docstring):
    """Extract keyword argument documentation from a function's docstring.

    Parameters
    ----------
    docstring: str
        The docstring to extract keyword arguments from.

    Returns
    -------
    list of (str, str, list str)

    str
        The name of the keyword argument.
    str
        Its type.
    str
        Its documentation as a list of lines.

    Notes
    -----
    The implementation is rather fragile.  It expects the following:

    1. The parameters are under an underlined Parameters section
    2. Keyword parameters have the literal ", optional" after the type
    3. Names and types are not indented
    4. Descriptions are indented with 4 spaces
    5. The Parameters section ends with an empty line.

    Examples
    --------

    >>> docstring = '''The foo function.
    ... Parameters
    ... ----------
    ... bar: str, optional
    ...     This parameter is the bar.
    ... baz: int, optional
    ...     This parameter is the baz.
    ...
    ... '''
    >>> kwargs = extract_kwargs(docstring)
    >>> kwargs[0]
    ('bar', 'str, optional', ['This parameter is the bar.'])

    """
    lines = inspect.cleandoc(docstring).split('\n')
    retval = []

    #
    # 1. Find the underlined 'Parameters' section
    # 2. Once there, continue parsing parameters until we hit an empty line
    #
    while lines[0] != 'Parameters':
        lines.pop(0)
    lines.pop(0)
    lines.pop(0)

    while lines and lines[0]:
        name, type_ = lines.pop(0).split(':', 1)
        description = []
        while lines and lines[0].startswith('    '):
            description.append(lines.pop(0).strip())
        if 'optional' in type_:
            retval.append((name.strip(), type_.strip(), description))

    return retval


def to_docstring(kwargs, lpad=''):
    """Reconstruct a docstring from keyword argument info.

    Basically reverses :func:`extract_kwargs`.

    Parameters
    ----------
    kwargs: list
        Output from the extract_kwargs function
    lpad: str, optional
        Padding string (from the left).

    Returns
    -------
    str
        The docstring snippet documenting the keyword arguments.

    Examples
    --------

    >>> kwargs = [
    ...     ('bar', 'str, optional', ['This parameter is the bar.']),
    ...     ('baz', 'int, optional', ['This parameter is the baz.']),
    ... ]
    >>> print(to_docstring(kwargs), end='')
    bar: str, optional
        This parameter is the bar.
    baz: int, optional
        This parameter is the baz.

    """
    buf = io.StringIO()
    for name, type_, description in kwargs:
        buf.write('%s%s: %s\n' % (lpad, name, type_))
        for line in description:
            buf.write('%s    %s\n' % (lpad, line))
    return buf.getvalue()
