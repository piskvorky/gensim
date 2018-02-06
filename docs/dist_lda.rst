.. _dist_lda:

Distributed Latent Dirichlet Allocation
============================================


.. note::
  See :doc:`distributed` for an introduction to distributed computing in `gensim`.


Setting up the cluster
_______________________

See the tutorial on :doc:`dist_lsi`; setting up a cluster for LDA is completely
analogous, except you want to run `lda_worker` and `lda_dispatcher` scripts instead
of `lsi_worker` and `lsi_dispatcher`.

Running LDA
____________

Run LDA like you normally would, but turn on the `distributed=True` constructor
parameter::

    >>> # extract 100 LDA topics, using default parameters
    >>> lda = LdaModel(corpus=mm, id2word=id2word, num_topics=100, distributed=True)
    using distributed version with 4 workers
    running online LDA training, 100 topics, 1 passes over the supplied corpus of 3199665 documets, updating model once every 40000 documents
    ..


In serial mode (no distribution), creating this online LDA :doc:`model of Wikipedia <wiki>`
takes 10h56m on my laptop (OS X, C2D 2.53GHz, 4GB RAM with `libVec`).
In distributed mode with four workers (Linux, Xeons of 2Ghz, 4GB RAM
with `ATLAS <http://math-atlas.sourceforge.net/>`_), the wallclock time taken drops to 3h20m.

To run standard batch LDA (no online updates of mini-batches) instead, you would similarly
call::

    >>> lda = LdaModel(corpus=mm, id2word=id2token, num_topics=100, update_every=0, passes=20, distributed=True)
    using distributed version with 4 workers
    running batch LDA training, 100 topics, 20 passes over the supplied corpus of 3199665 documets, updating model once every 3199665 documents
    initializing workers
    iteration 0, dispatching documents up to #10000/3199665
    iteration 0, dispatching documents up to #20000/3199665
    ...

and then, some two days later::

    iteration 19, dispatching documents up to #3190000/3199665
    iteration 19, dispatching documents up to #3199665/3199665
    reached the end of input; now waiting for all remaining jobs to finish

::

    >>> lda.print_topics(20)
    topic #0: 0.007*disease + 0.006*medical + 0.005*treatment + 0.005*cells + 0.005*cell + 0.005*cancer + 0.005*health + 0.005*blood + 0.004*patients + 0.004*drug
    topic #1: 0.024*king + 0.013*ii + 0.013*prince + 0.013*emperor + 0.008*duke + 0.008*empire + 0.007*son + 0.007*china + 0.007*dynasty + 0.007*iii
    topic #2: 0.031*film + 0.017*films + 0.005*movie + 0.005*directed + 0.004*man + 0.004*episode + 0.003*character + 0.003*cast + 0.003*father + 0.003*mother
    topic #3: 0.022*user + 0.012*edit + 0.009*wikipedia + 0.007*block + 0.007*my + 0.007*here + 0.007*edits + 0.007*blocked + 0.006*revert + 0.006*me
    topic #4: 0.045*air + 0.026*aircraft + 0.021*force + 0.018*airport + 0.011*squadron + 0.010*flight + 0.010*military + 0.008*wing + 0.007*aviation + 0.007*f
    topic #5: 0.025*sun + 0.022*star + 0.018*moon + 0.015*light + 0.013*stars + 0.012*planet + 0.011*camera + 0.010*mm + 0.009*earth + 0.008*lens
    topic #6: 0.037*radio + 0.026*station + 0.022*fm + 0.014*news + 0.014*stations + 0.014*channel + 0.013*am + 0.013*racing + 0.011*tv + 0.010*broadcasting
    topic #7: 0.122*image + 0.099*jpg + 0.046*file + 0.038*uploaded + 0.024*png + 0.014*contribs + 0.013*notify + 0.013*logs + 0.013*picture + 0.013*flag
    topic #8: 0.036*russian + 0.030*soviet + 0.028*polish + 0.024*poland + 0.022*russia + 0.013*union + 0.012*czech + 0.011*republic + 0.011*moscow + 0.010*finland
    topic #9: 0.031*language + 0.014*word + 0.013*languages + 0.009*term + 0.009*words + 0.008*example + 0.007*names + 0.007*meaning + 0.006*latin + 0.006*form
    topic #10: 0.029*w + 0.029*toronto + 0.023*l + 0.020*hockey + 0.019*nhl + 0.014*ontario + 0.012*calgary + 0.011*edmonton + 0.011*hamilton + 0.010*season
    topic #11: 0.110*wikipedia + 0.110*articles + 0.030*library + 0.029*wikiproject + 0.028*project + 0.019*data + 0.016*archives + 0.012*needing + 0.009*reference + 0.009*statements
    topic #12: 0.032*http + 0.030*your + 0.022*request + 0.017*sources + 0.016*archived + 0.016*modify + 0.015*changes + 0.015*creation + 0.014*www + 0.013*try
    topic #13: 0.011*your + 0.010*my + 0.009*we + 0.008*don + 0.008*get + 0.008*know + 0.007*me + 0.006*think + 0.006*question + 0.005*find
    topic #14: 0.073*r + 0.066*japanese + 0.062*japan + 0.018*tokyo + 0.008*prefecture + 0.005*osaka + 0.004*j + 0.004*sf + 0.003*kyoto + 0.003*manga
    topic #15: 0.045*da + 0.045*fr + 0.027*kategori + 0.026*pl + 0.024*nl + 0.021*pt + 0.017*en + 0.015*categoria + 0.014*es + 0.012*kategorie
    topic #16: 0.010*death + 0.005*died + 0.005*father + 0.004*said + 0.004*himself + 0.004*took + 0.004*son + 0.004*killed + 0.003*murder + 0.003*wife
    topic #17: 0.027*book + 0.021*published + 0.020*books + 0.014*isbn + 0.010*author + 0.010*magazine + 0.009*press + 0.009*novel + 0.009*writers + 0.008*story
    topic #18: 0.027*football + 0.024*players + 0.023*cup + 0.019*club + 0.017*fc + 0.017*footballers + 0.017*league + 0.011*season + 0.007*teams + 0.007*goals
    topic #19: 0.032*band + 0.024*album + 0.014*albums + 0.013*guitar + 0.013*rock + 0.011*records + 0.011*vocals + 0.009*live + 0.008*bass + 0.008*track



If you used the distributed LDA implementation in `gensim`, please let me know (my
email is at the bottom of this page). I would like to hear about your application and
the possible (inevitable?) issues that you encountered, to improve `gensim` in the future.
