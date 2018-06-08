# How to reproduce your benchmark
This document will explain the newly introduced files, how they are to be used and how to reproduce my benchmarks.

But before all that, here are my benchmarks. This output was generated and saved as a .csv using the script `evaluate_models.py`:

Method | map | ndcg@1 | ndcg@3 | ndcg@5 | ndcg@10 | ndcg@20
-- | -- | -- | -- | -- | -- | --
d2v | 0.406296 | 0.240664 | 0.373732 | 0.46163 | 0.531043 | 0.557495
w2v | 0.582535 | 0.443983 | 0.589505 | 0.643119 | 0.684337 | 0.701242
anmm | 0.60231 | 0.476793 | 0.617116 | 0.659008 | 0.710042 | 0.71779
arci | 0.568631 | 0.451477 | 0.563715 | 0.637976 | 0.672469 | 0.690647
cdssm | 0.523825 | 0.341772 | 0.54126 | 0.596421 | 0.638492 | 0.656304
conv_knrm_ranking | 0.569223 | 0.43038 | 0.587006 | 0.636489 | 0.684447 | 0.692889
drmm | 0.587125 | 0.447257 | 0.597009 | 0.646743 | 0.692769 | 0.706267
drmm_tks | **0.631183** | **0.506329** | **0.641376** | **0.690293** | **0.727095** | **0.73819**
dssm | 0.526801 | 0.341772 | 0.537857 | 0.597434 | 0.645009 | 0.656767
duet | 0.592516 | 0.451477 | 0.608629 | 0.657476 | 0.701125 | 0.711556
knrm_ranking | 0.501507 | 0.35865 | 0.499058 | 0.559678 | 0.619065 | 0.636557
matchpyramid | 0.605238 | 0.447257 | 0.62509 | 0.676466 | 0.714582 | 0.721466
mvlstm | 0.585831 | 0.459916 | 0.597372 | 0.648699 | 0.692848 | 0.705998


## Additional dependencies:
Unfortunately, the current state of the code needs the additional dependency of pandas, a module for hadnling .csv, .tsv, etc.
I was using it for grouping the datapoints by the document id. There are ways to do it without it and will be pushed soon.

So, you will have to install pandas first by running the command:
`pip install pandas`

## Current Directory Structure in gensim/similarity_learning
```.
.
├── HowToRunModels.md
├── HowToReproduceBenchmarks.md
├── __init__.py
├── custom_callbacks.py
├── custom_losses.py
├── output_log_mse.txt
├── output_log_rank_hing_loss.txt
├── dssm_example.py
├── evaluation_metrics.py
├── model.png
├── data
│   └── get_data.py
├── evaluation_scripts
│   ├── evaluate_models.py
│   ├── mz_results
│   │   ├── predict.test.anmm.wikiqa.txt
│   │   ├── predict.test.arci.wikiqa.txt
│   │   ├── predict.test.cdssm.wikiqa.txt
│   │   ├── predict.test.conv_knrm_ranking.wikiqa.txt
│   │   ├── predict.test.drmm_tks.wikiqa.txt
│   │   ├── predict.test.drmm.wikiqa.txt
│   │   ├── predict.test.dssm.wikiqa.txt
│   │   ├── predict.test.duet.wikiqa.txt
│   │   ├── predict.test.knrm_ranking.wikiqa.txt
│   │   ├── predict.test.matchpyramid.wikiqa.txt
│   │   └── predict.test.mvlstm.wikiqa.txt
├── models
│   ├── __init__.py
│   ├── drmm_tks.py
│   ├── dssm.py
└───├─── preprocessing
    ├── __init__.py
    ├── list_generator.py
    └── sl_vocab.py
```

For reproducing benchmarks only, we can ignore everything except the contents of folders "evaluation_scripts" and "data"

## Getting the data
Before we can run our evaluation script, we need to download the data set
This can be done through the `get_data.py` script in `gensim/similarity_learning/data/`

It is a utility script to download and unzip the datsets for Similarity Learning
It currently supports:
- WikiQA
- Quora Duplicate Question Pairs
- Glove 6 Billion tokens Word Embeddings

To get wikiqa
`$ python get_data.py --datafile wikiqa`

To get quoraqp
`$ python get_data.py --datafile quoraqp`

To get Glove Word Embeddings
`$ python get_data.py --datafile glove`

Note: 
- the evaluation scripts don't use QuoraQP yet

## Running evaluations:
The script for running evaluations is `evaluate_models.py` which can be found in `gensim/similarity_learning/evaluation_scripts/`
This script should be run to get a model-by-model or all-models evaluation. The script will evaluate the models based on their outputs to the "Train" split and save the results in a CSV if so prompted.

When running benchmarks on MatchZoo, we need to enter the output files produced by MatchZoo when predicting on the test dataset. MatchZoo provides a file in the format `predict.test.wikiqa.txt`.
In this case, I have collected my outputs and put them in `gensim/similarity_learning/evaluation_scripts/mz_results/` and renamed to include the name of the model used to generate it. So, `predict.test.wikiqa.txt` becomes `predict.test.model_name.wikiqa.txt`

Unfortunately, you will have to run [MatchZoo](https://github.com/faneshion/MatchZoo) and get the outputs yourself. For now, you can trust the results I have uploaded.


The script has several parameters which are best understood through the `--help`

```
usage: evaluate_models.py [-h] [--model MODEL] [--datapath DATAPATH]
                          [--word_embedding_path WORD_EMBEDDING_PATH]
                          [--mz_result_file MZ_RESULT_FILE]
                          [--result_save_path RESULT_SAVE_PATH]
                          [--mz_result_folder MZ_RESULT_FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         runs the evaluation on the given model type. Options
                        are: drmm_tks, doc2vec, word2vec, mz_eval,
                        mz_eval_multiple
  --datapath DATAPATH   path to the folder with WikiQACorpus. Path should
                        include WikiQACorpus Make sure you have run
                        get_data.py in gensim/similarity_learning/data/
  --word_embedding_path WORD_EMBEDDING_PATH
                        path to the Glove word embedding file
  --mz_result_file MZ_RESULT_FILE
                        path to the prediction output file made by mz
  --result_save_path RESULT_SAVE_PATH
                        path to save the results to as a csv
  --mz_result_folder MZ_RESULT_FOLDER
                        path to mz folder with many test prediction outputs
```



### Example usage:

For evaluating drmm_tks model on the WikiQA corpus
$ python evaluate_models.py --model drmm_tks --datapath ../data/WikiQACorpus/ --word_embedding_path ../evaluation_scripts/glove.6B.50d.txt --result_save_path results_drmm_tks

For evaluating doc2vec on the WikiQA corpus

`$ python evaluate_models.py --model doc2vec --datapath ../data/WikiQACorpus/`

For evaluating word2vec averaging on the WikiQA corpus

`$ python evaluate_models.py --model word2vec --datapath ../data/WikiQACorpus/ --word_embedding_path ../evaluation_scripts/glove.6B.50d.txt`

For evaluating the TREC format file produced by MatchZoo:

`$ python evaluate_models.py  --model mz --mz_result_file predict.test.wikiqa.txt`

Note: here "predict.test.wikiqa.txt" is the file output by MZ. It has been provided in this repo as an example.

For evaluating all models

with one mz output file

`$ python evaluate_models.py --model all --mz_result_file predict.test.wikiqa.txtDRMM --result_save_path results --word_embedding_path ../evaluation_scripts/glove.6B.50d.txt --datapath ../data/WikiQACorpus/`

-with a mz folder filled with result files

`$ python evaluate_models.py  --model all --mz_result_folder mz_results/ --result_save_path results_all --datapath ../data/WikiQACorpus/ --word_embedding_path ../evaluation_scripts/glove.6B.50d.txt`

## Description of Methods being evaluated

In all of the methods evaluated, I calcualte the metrics:
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (nDCG) at 1, 3, 5, 10, 20 rank length

### Word2Vec averaging
The word embeddings for each word in a query or document are taken and averaged.
The average is the representation of that query.
Eg: `"I am a king" -> mean(vec("I") ,vec("am"), vec("a"), vec("king")) :: an n dimensional vector`

For a given query `q1`, and a set of documents `(d1, d2, ...)`, the cosine similarity of `q1` and each doc is the similarity between the query and that doc.

The word embeddings are taken from the Glove Embeddings provided by [Stanford](https://nlp.stanford.edu/projects/glove/)

### Doc2Vec evaluation
WikiQA dataset is provided in an already train, test and validate split. 
1. I first train the d2v model on the train set.
2. Then I go through the test set, making an infer vector for each query or document.
3. I take the cosine similarity and choose the predicted answer as the highest cosine similarity.
4. Similar to the w2v averaging, I calculate the metrics.

### MatchZoo Evaluation
Match Zoo as part of its data preperation, splits the data set into test, valid and test. When their evaluation is run, they store the results in a `.txt` in a format similar to [TREC](https://github.com/usnistgov/trec_eval)
TREC format : `<query ID> Q0 <Doc ID> <rank> <score> <run ID>`

The format used by MatchZoo is
`<query ID> Q0 <Doc ID> <Doc No.> <predicted_score> <model name> <actual_score>`

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

Once we have their outputs and the ground truths, I run it through my function for calculatuing map and ndcg_at_k and output the results.
