from gensim.models.wrappers.dl4j.dl4jwrapper import dl4jWrapper
d = dl4jWrapper("raw_sentences.txt",5,1,100,42,5)
d.train()
