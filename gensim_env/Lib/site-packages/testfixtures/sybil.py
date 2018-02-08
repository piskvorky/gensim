from __future__ import absolute_import

import os
import re
import textwrap

from sybil import Region
from testfixtures import diff

FILEBLOCK_START = re.compile(r'^\.\.\s*topic::?\s*(.+)\b', re.MULTILINE)
FILEBLOCK_END = re.compile(r'(\n\Z|\n(?=\S))')
CLASS = re.compile(r'\s+:class:\s*(read|write)-file')


class FileBlock(object):
    def __init__(self, path, content, action):
        self.path, self.content, self.action = path, content, action


class FileParser(object):
    """
    A `Sybil <http://sybil.readthedocs.io>`__ parser that
    parses certain ReST sections to read and write files in the
    configured :class:`TempDirectory`.

    :param name: This is the name of the :class:`TempDirectory` to use
                 in the Sybil test namespace.

    """
    def __init__(self, name):
        self.name = name

    def __call__(self, document):
        for start_match, end_match, source in document.find_region_sources(
            FILEBLOCK_START, FILEBLOCK_END
        ):
            lines = source.splitlines()
            class_ = CLASS.match(lines[1])
            if not class_:
                continue
            index = 3
            if lines[index].strip() == '::':
                index += 1
            source = textwrap.dedent('\n'.join(lines[index:])).lstrip()
            if source[-1] != '\n':
                source += '\n'

            parsed = FileBlock(
                path=start_match.group(1),
                content=source,
                action=class_.group(1)
            )

            yield Region(
                start_match.start(),
                end_match.end(),
                parsed,
                self.evaluate
            )

    def evaluate(self, example):
        block = example.parsed
        dir = example.namespace[self.name]
        if block.action == 'read':
            actual = dir.read(block.path, 'ascii').replace(os.linesep, '\n')
            if actual != block.content:
                return diff(
                    block.content,
                    actual,
                    'File %r, line %i:' % (example.path, example.line),
                    'Reading from "%s":' % dir.getpath(block.path)
                )
        if block.action == 'write':
            dir.write(block.path, block.content, 'ascii')
