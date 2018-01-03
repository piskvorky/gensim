#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8

def ft_hash(unicode string):
    cdef unsigned int h = 2166136261
    for c in string:
        h ^= ord(c)
        h *= 16777619
    return h
