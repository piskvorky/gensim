#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-


"""Tento soubor obsahuje pozadavky na zmenu zpusobu pocitani SVD (singular value 
decomposition) velkych, ridkych matic z Pythonu.
 
SVD je algoritmicky problem linearni algebry -- na vstupu je ridka 2d matice, 
na vystupu mnozina vlastnich cisel a pravych a levych vlastnich vektoru.

Cilem bakalarky neni resit tento numericky problem (je komplikovany a byl uz 
uspokojive vyresen). Cilem je naroubovat existujici reseni na Python tak, aby 
se dalo: 
1) jednoduse vyuzivat spolu s knihovnou scipy = obecny prinos
2) jednoduse vyuzivat v projektu DML-CZ = prakticky prinos a soucasne objektivni metrika splneni ukolu

Teoreticky popis SVD, nazorne tutorialy a priklady k SVD viz google.

=== Popis
Vnitrne se v projektu DML-CZ pouziva modul scipy.sparse, z knihovny scipy (pro python 2.x 
neni scipy soucast standardnich knihoven, musi se nainstalovat zvlast, napr. z 
repozitare).
Protoze se pouzivaji velke matice, resi se hodne uspora mista pri reprezentaci
matic v RAM. Proto se pouzivaji reprezentace z modulu scipy.sparse (napr. trida 
scipy.sparse.csc_matrix), ktere do pameti ukladaji jen nenulove prvky matice 
( = "ridka (reprezentace) matice" = "sparse matrix"). 

Problem je, ze balik scipy poskytuje metody pro SVD pouze nad plnymi maticemi, ne 
nad ridkymi, viz scipy.linalg.svd. Pro SVD nad ridkymi maticemi se proto pouzivaji
dalsi knihovny, obvykle napsane v C, C++ nebo ve FORTRANu (svdlibc, svdpack, propack, ...).
Nastava tak prakticky problem s rozhranim mezi scipy a touto externi knihovnou.

=== Soucasne reseni
Pouzivam externi knihovnu divisi (http://divisi.media.mit.edu/), coz je 
Pythonovske zabaleni Ckove knihovny svdlibc. 
Problem je, ze pro reprezentaci ridkych matic v RAM pouziva vlastni strukturu --
tridu divisi.DictTensor(2). Pro ziskani SVD dekompozice se proto musi
provest nasledujici harakiri:
1) Python objekty z DML-CZ projektu (matice typu scipy.sparse.csc_matrix) se 
   prevedou na Python objekty typu divisi.DictTensor(2).
2) matice typu divisi.DictTensor(2) se interne prevede na format pouzivany v svdlibc
3) pusti se svdlibc
4) vysledky svdlibc se prevedou na typ divisi.DictTensor(2)
5) vysledky v divisi.DictTensor(2) se prevedou na typ scipy.sparse.csc_matrix

Dole prikladam kod ke stavajicimu reseni. Jde o funkce primo vykopirovane ze 
soucasneho reseni v DML-CZ. Pro spusteni je potreba mit nainstalovany python2.5, 
divisi, numpy, scipy atd.

=== Pozadovane reseni
Zbavit se mezikroku s divisi knihovnou. Tzn. prevadet scipy.sparse.csc_matrix primo 
na vnitrni format zvolene knihovny, pustit SVD, vysledky prevest zpet na csc_matrix.

Pointa je, ze nebude potreba udrzovat v RAM jednu kopii matice zbytecne navic. Zatimco 
ted je matice v bode 3) v pameti trikrat (1x scipy.sparse, 1x divisi.Tensor, 1x 
interni libsvdc reprezentace), nove bude nejvyse 2x (1x scipy.sparse.csc_matrix, 
1x vnitrni reprezentace ve zvolene knihovne).

Pri tvorbe obalu zvolene knihovny se lze pochopitelne s vyhodou inspirovat zpusobem, 
jakym zabaleni resili napr. v divisi.

=== Format vystupu
Praktickym vystupem prace bude (krome textu bakalarky) hlavne instalacni skript, 
ktery vytvorenou knihovnu kompletne nainstaluje, idealne pak debian balicek
vcetne zavislosti. 
Opet se lze inspirovat instalaci tak, jak je resena v divisi.

Knihovnu bude po instalaci mozne pouzivat jednoduse jako:

# import jmeno_nove_svd_knihovny
# u, s, vt = jmeno_nove_svd_knihovny.doSvd(matice ve formatu csc_matrix, pocet pozadovanych vl. cisel)

kde u, s, a vt jsou typu standardni numpy pole, viz taky fce doSvd nize.
"""

import logging

import numpy
import scipy
import scipy.io.mmio

from csc import divisi


def iterateCsc(mat):
    """
    Iterate over CSC matrix, returning (key, value) pairs, where key = (row, column) 2-tuple.
    Ie. simulate mat.iteritems() as it should have been in scipy.sparse...
    
    Depends on scipy.sparse.csc_matrix implementation details!
    """
    if not isinstance(mat, scipy.sparse.csc_matrix):
        raise TypeError("iterateCsc expects an CSC matrix on input!")
    for col in xrange(mat.shape[1]):
        if col % 1000 == 0:
            logging.debug("iterating over column %i/%i" % (col, mat.shape[1]))
        for i in xrange(mat.indptr[col], mat.indptr[col + 1]):
            row, value = mat.indices[i], mat.data[i]
            yield (row, col), value


def toTensor(sparseMat):
    """
    Convert scipy.sparse matrix into divisi.DictTensor matrix.
    
    Accepts any scipy.sparse representation which can be converted to scipy.sparse.csc_matrix.
    """
    logging.info("creating divisi sparse tensor of shape %s" % (sparseMat.shape,))
    sparseTensor = divisi.DictTensor(ndim = 2)
    sparseTensor.update(iterateCsc(sparseMat.tocsc())) # first convert to csc_matrix, then fill tensor with data
    return sparseTensor


def doSVD(sparseMat, num):
    """
    Do Singular Value Decomposition on sparseMat using an external program (SVDLIBC via divisi).
     
    mat is a sparse matrix in any scipy.sparse format, num is the number of requested eigenvectors.
    
    Return the 3-tuple (U, S, VT) = (left eigenvectors, singular values, right eigenvectors)
    """
    logging.info("computing sparse svd of %s matrix" % (sparseMat.shape,))
    svdResult = divisi.svd.svd_sparse(toTensor(sparseMat), k = num) # FIXME here, at least three copies of the original sparseMat exist in memory!!
    
    # convert sparse tensors (result of sparse_svd in divisi) back to standard numpy arrays
    u = svdResult.u.unwrap()
    v = svdResult.v.unwrap()
    s = svdResult.svals.unwrap()
    logging.info("result of svd: u=%s, s=%s, v=%s" % (u.shape, s.shape, v.shape))
    assert len(s) <= num # make sure we didn't get more than we asked for
    assert len(s) == u.shape[1] == v.shape[1] # make sure the dimensions fit
    return u, s, v.T

# ==================================================================
# priklad vytvoreni miniaturni matice a provedeni SVD pomoci divisi.
# ==================================================================
if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG)

    import numpy.random
    mat = numpy.random.random((20, 10)) # create 20x10 matrix filled with random elements
    sparseMat = scipy.sparse.lil_matrix(mat) # convert to sparse representation
    print 'input matrix:', repr(mat)
    
    NUMDIM = 5 # number of required eigenvalues
    u, s, vt = doSVD(sparseMat, NUMDIM)
    print 'S:', s
    print 'U:', u.shape, u
    print 'V^T:', vt.shape, vt

