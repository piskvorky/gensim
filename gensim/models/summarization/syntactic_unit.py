#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


class SyntacticUnit(object):

    def __init__(self, text, token=None, tag=None):
        self.text = text
        self.token = token
        self.tag = tag[:2] if tag else None     # Just first two letters of tag
        self.index = -1
        self.score = -1

    def __str__(self):
        return "Original unit: '" + self.text + "' *-*-*-* " + "Processed unit: '" + self.token + "'"

    def __repr__(self):
        return str(self)
