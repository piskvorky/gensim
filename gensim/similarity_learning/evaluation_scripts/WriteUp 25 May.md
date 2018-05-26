# Evaluation of Models

## word2vec averaging evaluation
The word embeddings for each word in a query are taken and averaged.
The average is the representation of that query.
Eg: `"I am a king" -> mean(vec("I") ,vec("am"), vec("a"), vec("king")) :: an n dimensional vector`

For a given query q1, and a set of documents (d1, d2, ...),
the cosine similarity of q1 and each doc is the similarity between the query and that doc.

The word embeddings are taken from the Glove Embeddings provided by [Stanford](https://nlp.stanford.edu/projects/glove/)

**File :** [w2v_avg_eval.py](w2v_avg_eval.py)

```
Accuracy :  0.2867264997638167
map : 0.525039852586227
ndcg@ 1  :  0.37844036697247707
ndcg@ 3  :  0.5288832435950492
ndcg@ 5  :  0.5938376161901073
ndcg@ 10  :  0.6426858123695702
ndcg@ 20  :  0.6578442782634695
```

## doc2vec evaluation
WikiQA dataset is provided in an already train, test, validate split.
I first train the d2v model on the train set.
Then I go through the test set, making an infer vector for each query document pair. I take the cosine_similarity
and choose the predicted answer as the highest cosine similarity.
Similar to the w2v averaging, I calculate the metrics.

**File :** [d2v_train_eval.py](d2v_train_eval.py)

```
map : 0.3694044782081827
ndcg@ 1  :  0.1908713692946058
ndcg@ 3  :  0.33425007107376825
ndcg@ 5  :  0.4149963330213527
ndcg@ 10  :  0.49378684260187083
ndcg@ 20  :  0.5293908121349118
```

## MatchZoo Evaluation
Match Zoo as part of its data preperation, splits the data set intpo test, valid and test. [TODO add link]
When their evaluation is run, they store the results in a txt in a format similar to TREC
TREC format : `<query ID> Q0 <Doc ID> <rank> <score> <run ID>`

The format used by MatchZoo is
```
<query ID> Q0 <Doc ID> <Doc No.> <predicted_score> <model name> <actual_score>
```

Example output for the DRMM model
```
Q2241	Q0	D19687	0	7.481709	DRMM	1
Q2241	Q0	D19684	1	7.031933	DRMM	0
Q2241	Q0	D19682	2	6.355274	DRMM	0
Q2241	Q0	D19686	3	5.907663	DRMM	0
Q2241	Q0	D19685	4	5.899095	DRMM	0
Q2241	Q0	D19683	5	5.824327	DRMM	0
Q2242	Q0	D19690	0	6.724691	DRMM	0
Q2242	Q0	D19693	1	5.881740	DRMM	0
```

Note: it is unclear as to why Q0 is written.

So, for the MatchZoo evaluation, my script goes through this file output by MZ.
It then groups it based on Question number.
For example:
```
y_pred = [
		  [predicted(Q1, D1), predicted(Q1, D2), predicted(Q1, D3)], 
		  [predicted(Q2,D1), predicted(Q2, D2]], 
		  ... 
		 ]

y_true = [
		  [true_val(Q1, D1), true_val(Q1, D2), true_val(Q1, D3)], 
		  [true_val(Q2,D1), true_val(Q2, D2]],
		  ... 
		 ]
```

Once we have their outputs and the ground truths, I run it through my function for map and ndcg_at_k
and output the results.

**File used :** [mz_eval.py](mz_eval.py)
