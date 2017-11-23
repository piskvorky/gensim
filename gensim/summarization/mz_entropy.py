#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


from gensim.summarization.textcleaner import tokenize_by_word as _tokenize_by_word
from gensim.utils import to_unicode
import numpy
import scipy

def mz_keywords(text,blocksize=1024,scores=False,split=False,weighted=True,threshold=0.0):
    """Extract keywords from text using the Montemurro and Zanette entropy algorithm.
       https://arxiv.org/abs/0907.1558
       :param text: str (document to summarize)
       :param blocksize: int (size of blocks to use in analysis)
       :params scores: bool (return score with keywords)
       :params split: bool (return results as list)
       :params weighted: bool (weight scores by word frequency)
       :params threshold: float or 'auto' (minimum score for returned keywords)"""
    text=to_unicode(text)
    words=_tokenize_by_word(text)
    vocab=sorted(set(words))
    wordcounts=numpy.array([[words[i:i+blocksize].count(word) for word in vocab]
                            for i in range(0,len(words),blocksize)])
    nblocks=wordcounts.shape[0]
    totals=wordcounts.sum(axis=0)
    nwords=totals.sum()
    p=wordcounts/totals
    logp=numpy.nan_to_num(numpy.log2(p),0.0)
    H=logp.sum(axis=0)
    
    def log_combinations(n,m):
        """Calculates the logarithm of n!/m!(n-m)!"""
        return -(numpy.log(n+1)+scipy.special.betaln(n-m+1,m+1))
    
    def marginal_prob(n,m):
        """Marginal probability of a word that occurs n times in the document
           occurring m times in a given block"""
        return numpy.exp(log_combinations(n,m)
                         +log_combinations(nwords-n,blocksize-m)
                         -log_combinations(nwords,blocksize))
        
    marginal=numpy.frompyfunc(marginal_prob,2,1)
        
    def analytic_entropy(n):
        """Predicted entropy for a word that occurs n times in the document"""
        m=numpy.arange(1,min(blocksize,n)+1)
        p=m/n
        elements=p*numpy.nan_to_num(numpy.log2(p))*marginal(n,m)
        return -nblocks*elements.sum()
    
    analytic=numpy.frompyfunc(analytic_entropy,1,1)
    
    H+=analytic(totals)
    if weighted:
        H*=totals/nwords
    if threshold=='auto':
        threshold=nblocks/(nblocks+1.0)
    weights=[(word,score) 
             for (word,score) in zip(vocab,H)
             if score>threshold]
    weights.sort(key=lambda x:-x[1])
    result= weights if scores else [word for (word,score) in weights]
    if not (scores or split):
        result='\n'.join(result)
    return result
        
    