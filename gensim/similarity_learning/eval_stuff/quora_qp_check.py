from gensim.similarity_learning import QuoraQPExtractor
import os, pickle
import numpy as np

quoraqp = QuoraQPExtractor(os.path.join("..","data", "QuoraQP", "quora_duplicate_questions.tsv"))
qp, labels = quoraqp.get_data()


with open('w2v.pkl', 'rb') as f:
    w2v = pickle.load(f)

def sent2vec(sentence):
    vec_sum = []

    for word in sentence.split():
        if word in w2v:
            vec_sum.append(w2v[word])

    return np.mean(np.array(vec_sum), axis=0)

def cos_sim(vec1, vec2):
    return np.sum(vec1*vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

# print(labels)
n_duplicates = np.sum(np.array(labels))
perc_dupl = n_duplicates/len(labels)*100
print('There are %d duplicate questions of the %d total questions, i.e., %f%%' % (n_duplicates, len(labels), perc_dupl))

for (q1, q2), l in zip(qp, labels):
	q1_vec = sent2vec(q1)
	q2_vec = sent2vec(q2)
	break
	# print(cos_sim(q1_vec, q2_vec))