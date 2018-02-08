from gensim.models import Phrases 
#1st way to get error
bigram_model = Phrases()

#2nd way to get error
bigram_model = Phrases(['aaa', 'aaa'])