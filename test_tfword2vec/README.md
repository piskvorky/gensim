# Run tests for distributed word2vec

Run the following commands in test_tfword2vec directory.

Install data for testing and evaluating:

```
curl http://mattmahoney.net/dc/text8.zip > text8.zip
curl https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip > source-archive.zip
unzip -p source-archive.zip  word2vec/trunk/questions-words.txt > questions-words.txt
rm source-archive.zip
```

Add dir for saving models after training:

```
mkdir models
```

Building:

```
docker-compose build
```

Run Jupyter Notebook, go to localhost:8888 and open Test_TfWord2Vec.ipynb:

```
jupyter notebook
```


