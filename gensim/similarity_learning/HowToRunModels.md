# Running Similarity Learning Models

## DRMM_TKS (Deep Relevance Matching Model Top k Solutions)
Example usage:

`$ python drmm_tks_example.py --wikiqa_folder_path ./data/WikiQACorpus/ --word_embedding_path evaluation_scripts/glove.6B.50d.txt`

This script will make a WikiQA_DRMM_TKS_Extractor and DRMM_TKS object and train it.

**Link to Paper:** [A Deep Relevance Matching Model for Ad-hoc Retrieval](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf)

Refer to the image below to understand the network topology. <br/>Note: The image boxes are labelled so:
`a = Dense(10)(b) -> a_Dense_10_b`

![DRMM_TKS_Topology](model.png)

## DSSM (Deep Structured Semantic Model)
Example Usage:
`$ python dssm_example.py --wikiqa_folder_path ./data/WikiQACorpus/`

Note: There's a bug in the WikiQAExtractor for DSSM. Might lead to minor problems. Will be fixed/reimplemented when we get to that model.

**Link to Paper:** [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)