Changes
===========
## 2.2.0, 2017-06-21


:star2: New features:
* Add sklearn wrapper for RpModel (@chinmayapancholi13, [#1395](https://github.com/RaRe-Technologies/gensim/pull/1395))
* Add sklearn wrappers for LdaModel and LsiModel (@chinmayapancholi13, [#1398](https://github.com/RaRe-Technologies/gensim/pull/1398))
* Add sklearn wrapper for LdaSeq (@chinmayapancholi13, [#1405](https://github.com/RaRe-Technologies/gensim/pull/1405))
* Add keras wrapper for Word2Vec model (@chinmayapancholi13, [#1248](https://github.com/RaRe-Technologies/gensim/pull/1248))
* Add LdaModel.diff method (@menshikh-iv, [#1334](https://github.com/RaRe-Technologies/gensim/pull/1334))
* Allow use of truncated Dictionary for coherence measures. Fix #1342 (@macks22, [#1349](https://github.com/RaRe-Technologies/gensim/pull/1349))


:+1: Improvements:
* Fix save_as_text/load_as_text for Dictionary (@vlejd, [#1402](https://github.com/RaRe-Technologies/gensim/pull/1402))
* Add sampling support for corpus. Fix #308 (@vlejd, [#1408](https://github.com/RaRe-Technologies/gensim/pull/1408))
* Add napoleon extension to sphinx (@rasto2211, [#1411](https://github.com/RaRe-Technologies/gensim/pull/1411))
* Add KeyedVectors support to AnnoyIndexer (@quole, [#1318](https://github.com/RaRe-Technologies/gensim/pull/1318))
* Add BaseSklearnWrapper (@chinmayapancholi13, [#1383](https://github.com/RaRe-Technologies/gensim/pull/1383))
* Replace num_words to topn in model for unification. Fix #1198 (@prakhar2b, [#1200](https://github.com/RaRe-Technologies/gensim/pull/1200))
* Rename out_path to out_name & add logging for WordRank model. Fix #1310 (@parulsethi, [#1332](https://github.com/RaRe-Technologies/gensim/pull/1332))
* Remove multiple iterations of corpus in p_boolean_document (@danielchamberlain, [#1325](https://github.com/RaRe-Technologies/gensim/pull/1325))
* Fix codestyle in TfIdf (@piskvorky, [#1313](https://github.com/RaRe-Technologies/gensim/pull/1313))
* Fix warnings from Sphinx. Partial fix #1192 (@souravsingh, [#1330](https://github.com/RaRe-Technologies/gensim/pull/1330))
* Add test_env to setup.py (@menshikh-iv, [#1336](https://github.com/RaRe-Technologies/gensim/pull/1336))


:red_circle: Bug fixes:
* Add cleanup in annoy test (@prakhar2b, [#1420](https://github.com/RaRe-Technologies/gensim/pull/1420))
* Add cleanup in lda backprop test (@prakhar2b, [#1417](https://github.com/RaRe-Technologies/gensim/pull/1417))
* Fix out-of-vocab in FastText (@jayantj, [#1409](https://github.com/RaRe-Technologies/gensim/pull/1409))
* Add cleanup in WordRank test (@parulsethi, [#1410](https://github.com/RaRe-Technologies/gensim/pull/1410))
* Fix rest requirements in Travis. Partial fix #1393 (@ibrahimsharaf, @menshikh-iv, [#1400](https://github.com/RaRe-Technologies/gensim/pull/1400))
* Fix morfessor exception. Partial fix #1324 (@souravsingh, [#1406](https://github.com/RaRe-Technologies/gensim/pull/1406))
* Fix test for FastText (@prakhar2b, [#1371](https://github.com/RaRe-Technologies/gensim/pull/1371))
* Fix WikiCorpus (@alekol, [#1333](https://github.com/RaRe-Technologies/gensim/pull/1333))
* Fix backward incompatibility for LdaModel (@chinmayapancholi13, [#1327](https://github.com/RaRe-Technologies/gensim/pull/1327))
* Fix support for old and new FastText model format. Fix #1301 (@prakhar2b, [#1319](https://github.com/RaRe-Technologies/gensim/pull/1319))
* Fix wrapper tests. Fix #1323 (@shubhamjain74, [#1359](https://github.com/RaRe-Technologies/gensim/pull/1359))
* Update export_phrases method. Fix #794 (@toumorokoshi, [#1362](https://github.com/RaRe-Technologies/gensim/pull/1362))
* Fix sklearn exception in test (@souravsingh, [#1350](https://github.com/RaRe-Technologies/gensim/pull/1350))


:books: Tutorial and doc improvements:
* Fix incorrect link in tutorials (@aneesh-joshi, [#1426](https://github.com/RaRe-Technologies/gensim/pull/1426))
* Add notebook with sklearn wrapper examples (@chinmayapancholi13, [#1428](https://github.com/RaRe-Technologies/gensim/pull/1428))
* Replace absolute pathes to relative in notebooks (@vochicong, [#1414](https://github.com/RaRe-Technologies/gensim/pull/1414))
* Fix code-style in keras notebook (@chinmayapancholi13, [#1394](https://github.com/RaRe-Technologies/gensim/pull/1394))
* Replace absolute pathes to relative in notebooks (@vochicong, [#1407](https://github.com/RaRe-Technologies/gensim/pull/1407))
* Fix typo in quickstart guide (@vochicong, [#1404](https://github.com/RaRe-Technologies/gensim/pull/1404))
* Update docstring for WordRank. Fix #1384 (@parulsethi, [#1378](https://github.com/RaRe-Technologies/gensim/pull/1378))
* Update docstring for SkLdaModel (@chinmayapancholi13, [#1382](https://github.com/RaRe-Technologies/gensim/pull/1382))
* Update logic for updatetype in LdaModel (@chinmayapancholi13, [#1389](https://github.com/RaRe-Technologies/gensim/pull/1389))
* Update docstring for Doc2Vec (@jstol, [#1379](https://github.com/RaRe-Technologies/gensim/pull/1379))
* Fix docstring for KL-distance (@viciousstar, [#1373](https://github.com/RaRe-Technologies/gensim/pull/1373))
* Update Corpora_and_Vector_Spaces tutorial (@charliejharrison, [#1308](https://github.com/RaRe-Technologies/gensim/pull/1308))
* Add visualization for difference between LdaModel (@menshikh-iv, [#1374](https://github.com/RaRe-Technologies/gensim/pull/1374))
* Fix punctuation & typo in changelog (@piskvorky, @menshikh-iv, [#1366](https://github.com/RaRe-Technologies/gensim/pull/1366))
* Fix PEP8 & typo in several PRs (@menshikh-iv, [#1369](https://github.com/RaRe-Technologies/gensim/pull/1369))
* Update docstrings connected with backward compability in for LdaModel (@chinmayapancholi13, [#1365](https://github.com/RaRe-Technologies/gensim/pull/1365))
* Update Corpora_and_Vector_Spaces tutorial (@schuyler1d, [#1360](https://github.com/RaRe-Technologies/gensim/pull/1360))
* Fix typo in Doc2Vec doctsring (@fujiyuu75, [#1356](https://github.com/RaRe-Technologies/gensim/pull/1356))
* Update Annoy tutorial (@pmbaumgartner, [#1355](https://github.com/RaRe-Technologies/gensim/pull/1355))
* Update temp folder in tutorials (@yl2526, [#1352](https://github.com/RaRe-Technologies/gensim/pull/1352))
* Remove spaces after print in Topics_and_Transformation tutorial (@gsimore, [#1354](https://github.com/RaRe-Technologies/gensim/pull/1354))
* Update Dictionary docstring (@oonska, [#1347](https://github.com/RaRe-Technologies/gensim/pull/1347))
* Add section headings in word2vec notebook (@MikeTheReader, [#1348](https://github.com/RaRe-Technologies/gensim/pull/1348))
* Fix broken urls in starter tutorials (@ka7eh, [#1346](https://github.com/RaRe-Technologies/gensim/pull/1346))
* Update quick start notebook (@yardsale8, [#1345](https://github.com/RaRe-Technologies/gensim/pull/1345))
* Fix typo in quick start notebook (@MikeTheReader, [#1344](https://github.com/RaRe-Technologies/gensim/pull/1344))
* Fix docstring in keyedvectors (@chinmayapancholi13, [#1337](https://github.com/RaRe-Technologies/gensim/pull/1337))



## 2.1.0, 2017-05-12

:star2: New features:
* Add modified save_word2vec_format for Doc2Vec, to save document vectors. (@parulsethi, [#1256](https://github.com/RaRe-Technologies/gensim/pull/1256))


:+1: Improvements:
* Add automatic code style check limited only to the code modified in PR (@tmylk, [#1287](https://github.com/RaRe-Technologies/gensim/pull/1287))
* Replace `logger.warn` by `logger.warning` (@chinmayapancholi13, [#1295](https://github.com/RaRe-Technologies/gensim/pull/1295))
* Docs word2vec docstring improvement, deprecation labels (@shubhvachher, [#1274](https://github.com/RaRe-Technologies/gensim/pull/1274))
* Stop passing 'sentences' as parameter to Doc2Vec. Fix #511 (@gogokaradjov, [#1306](https://github.com/RaRe-Technologies/gensim/pull/1306))


:red_circle: Bug fixes:
* Allow indexing with np.int64 in doc2vec. Fix #1231 (@bogdanteleaga, [#1254](https://github.com/RaRe-Technologies/gensim/pull/1254))
* Update Doc2Vec docstring. Fix #1302 (@datapythonista, [#1307](https://github.com/RaRe-Technologies/gensim/pull/1307))
* Ignore rst and ipynb file in Travis flake8 validations (@datapythonista, [#1309](https://github.com/RaRe-Technologies/gensim/pull/1309))


:books: Tutorial and doc improvements:
* Update Tensorboard Doc2Vec notebook (@parulsethi, [#1286](https://github.com/RaRe-Technologies/gensim/pull/1286))
* Update Doc2Vec IMDB Notebook, replace codesc to smart_open (@robotcator, [#1278](https://github.com/RaRe-Technologies/gensim/pull/1278))
* Add explanation of `size` to Word2Vec Notebook (@jbcoe, [#1305](https://github.com/RaRe-Technologies/gensim/pull/1305))
* Add extra param to WordRank notebook. Fix #1276 (@parulsethi, [#1300](https://github.com/RaRe-Technologies/gensim/pull/1300))
* Update warning message in WordRank (@parulsethi, [#1299](https://github.com/RaRe-Technologies/gensim/pull/1299))


## 2.0.0, 2017-04-10

Breaking changes:

Any direct calls to method train() of Word2Vec/Doc2Vec now require an explicit epochs parameter and explicit estimate of corpus size. The most usual way to call `train` is `vec_model.train(sentences, total_examples=self.corpus_count, epochs=self.iter)`
See the [method documentation](https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py#L766) for more information.


* Explicit epochs and corpus size in word2vec train(). (@gojomo, @robotcator, [#1139](https://github.com/RaRe-Technologies/gensim/pull/1139), [#1237](https://github.com/RaRe-Technologies/gensim/pull/1237))

New features:
* Add output word prediction in word2vec. Only for negative sampling scheme. See [ipynb]( https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/word2vec.ipynb) (@chinmayapancholi13, [#1209](https://github.com/RaRe-Technologies/gensim/pull/1209))
* scikit_learn wrapper for LSI Model in Gensim (@chinmayapancholi13, [#1244](https://github.com/RaRe-Technologies/gensim/pull/1244))
* Add the 'keep_tokens' parameter to 'filter_extremes'. (@toliwa, [#1210](https://github.com/RaRe-Technologies/gensim/pull/1210))
* Load FastText models with specified encoding (@jayantj, [#1210](https://github.com/RaRe-Technologies/gensim/pull/1189))


Improvements:
* Fix loading large FastText models on Mac. (@jaksmid, [#1196](https://github.com/RaRe-Technologies/gensim/pull/1214))
* Sklearn LDA wrapper now works in sklearn pipeline (@kris-singh, [#1213](https://github.com/RaRe-Technologies/gensim/pull/1213))
* glove2word2vec conversion script refactoring (@parulsethi, [#1247](https://github.com/RaRe-Technologies/gensim/pull/1247))
* Word2vec error message when update called before train . Fix #1162 (@hemavakade, [#1205](https://github.com/RaRe-Technologies/gensim/pull/1205))
* Allow training if model is not modified by "_minimize_model". Add deprecation warning. (@chinmayapancholi13, [#1207](https://github.com/RaRe-Technologies/gensim/pull/1207))
* Update the warning text when building vocab on a trained w2v model (@prakhar2b, [#1190](https://github.com/RaRe-Technologies/gensim/pull/1190))

Bug fixes:

*  Fix word2vec reset_from bug in v1.0.1 Fix #1230. (@Kreiswolke, [#1234](https://github.com/RaRe-Technologies/gensim/pull/1234))

* Distributed LDA: checking the length of docs instead of the boolean value, plus int index conversion (@saparina, [#1191](https://github.com/RaRe-Technologies/gensim/pull/1191))

* syn0_lockf initialised with zero in intersect_word2vec_format() (@KiddoZhu, [#1267](https://github.com/RaRe-Technologies/gensim/pull/1267))

* Fix wordrank max_iter_dump calculation. Fix #1216 (@ajkl, [#1217](https://github.com/RaRe-Technologies/gensim/pull/1217))

* Make SgNegative test use skip-gram (@shubhvachher, [#1252](https://github.com/RaRe-Technologies/gensim/pull/1252))

* pep8/pycodestyle fixes for hanging indents in Summarization module (@SamriddhiJain, [#1202](https://github.com/RaRe-Technologies/gensim/pull/1202))

* WordRank and Mallet wrappers single vs double quote issue in windows. (@prakhar2b, [#1208](https://github.com/RaRe-Technologies/gensim/pull/1208))


* Fix #824 : no corpus in init, but trim_rule in init (@prakhar2b, [#1186](https://github.com/RaRe-Technologies/gensim/pull/1186))

* Hardcode version number. Fix #1138. (@tmylk, [#1138](https://github.com/RaRe-Technologies/gensim/pull/1138))

Tutorial and doc improvements:

* Color dictionary according to topic notebook update (@bhargavvader, [#1164](https://github.com/RaRe-Technologies/gensim/pull/1164))

* Fix hdp show_topic/s docstring (@parulsethi, [#1264](https://github.com/RaRe-Technologies/gensim/pull/1264))

* Add docstrings for word2vec.py forwarding functions (@shubhvachher, [#1251](https://github.com/RaRe-Technologies/gensim/pull/1251))

* updated description for worker_loop function used in score function (@chinmayapancholi13, [#1206](https://github.com/RaRe-Technologies/gensim/pull/1206))

## 1.0.1, 2017-03-03

* Rebuild cumulative table on load. Fix #1180. (@tmylk, [#1181](https://github.com/RaRe-Technologies/gensim/pull/893))
* most_similar_cosmul bug fix (@dkim010, [#1177](https://github.com/RaRe-Technologies/gensim/pull/1177))
* Fix loading old word2vec models pre-1.0.0  (@jayantj, [#1179](https://github.com/RaRe-Technologies/gensim/pull/1179))
* Load utf-8 words in fasttext  (@jayantj, [#1176](https://github.com/RaRe-Technologies/gensim/pull/1176))


## 1.0.0, 2017-02-24

New features:
* Add Author-topic modeling (@olavurmortensen, [#893](https://github.com/RaRe-Technologies/gensim/pull/893))
* Add FastText word embedding wrapper (@Jayantj, [#847](https://github.com/RaRe-Technologies/gensim/pull/847))
* Add WordRank word embedding  wrapper (@parulsethi, [#1066](https://github.com/RaRe-Technologies/gensim/pull/1066), [#1125](https://github.com/RaRe-Technologies/gensim/pull/1125))
* Add VarEmbed word embedding wrapper (@anmol01gulati, [#1067](https://github.com/RaRe-Technologies/gensim/pull/1067)))
* Add sklearn wrapper for LDAModel (@AadityaJ, [#932](https://github.com/RaRe-Technologies/gensim/pull/932))

Deprecated features:

* Move `load_word2vec_format` and `save_word2vec_format` out of Word2Vec class to KeyedVectors (@tmylk, [#1107](https://github.com/RaRe-Technologies/gensim/pull/1107))
* Move properties `syn0norm`, `syn0`, `vocab`, `index2word` from Word2Vec class to KeyedVectors (@tmylk,[#1147](https://github.com/RaRe-Technologies/gensim/pull/1147))
* Remove support for Python 2.6, 3.3 and 3.4 (@tmylk,[#1145](https://github.com/RaRe-Technologies/gensim/pull/1145))


Improvements:

* Python 3.6 support (@tmylk [#1077](https://github.com/RaRe-Technologies/gensim/pull/1077))
* Phrases and Phraser allow a generator corpus (ELind77 [#1099](https://github.com/RaRe-Technologies/gensim/pull/1099))
* Ignore DocvecsArray.doctag_syn0norm in save. Fix #789 (@accraze, [#1053](https://github.com/RaRe-Technologies/gensim/pull/1053))
* Fix bug in LsiModel that occurs when id2word is a Python 3 dictionary. (@cvangysel, [#1103](https://github.com/RaRe-Technologies/gensim/pull/1103)
* Fix broken link to paper in readme (@bhargavvader, [#1101](https://github.com/RaRe-Technologies/gensim/pull/1101))
* Lazy formatting in evaluate_word_pairs (@akutuzov, [#1084](https://github.com/RaRe-Technologies/gensim/pull/1084))
* Deacc option to keywords pre-processing (@bhargavvader, [#1076](https://github.com/RaRe-Technologies/gensim/pull/1076))
* Generate Deprecated exception when using Word2Vec.load_word2vec_format (@tmylk, [#1165](https://github.com/RaRe-Technologies/gensim/pull/1165))
* Fix hdpmodel constructor docstring for print_topics (#1152) (@toliwa, [#1152](https://github.com/RaRe-Technologies/gensim/pull/1152))
* Default to per_word_topics=False in LDA get_item for performance (@menshikh-iv, [#1154](https://github.com/RaRe-Technologies/gensim/pull/1154))
* Fix bound computation in Author Topic models. (@olavurmortensen, [#1156](https://github.com/RaRe-Technologies/gensim/pull/1156))
* Write UTF-8 byte strings in tensorboard conversion (@tmylk, [#1144](https://github.com/RaRe-Technologies/gensim/pull/1144))
* Make top_topics and sparse2full compatible with numpy 1.12 strictly int idexing (@tmylk, [#1146](https://github.com/RaRe-Technologies/gensim/pull/1146))

Tutorial and doc improvements:

* Clarifying comment in is_corpus func in utils.py (@greninja, [#1109](https://github.com/RaRe-Technologies/gensim/pull/1109))
* Tutorial Topics_and_Transformations fix markdown and add references (@lgmoneda, [#1120](https://github.com/RaRe-Technologies/gensim/pull/1120))
* Fix doc2vec-lee.ipynb results to match previous behavior (@bahbbc, [#1119](https://github.com/RaRe-Technologies/gensim/pull/1119))
* Remove Pattern lib dependency in News Classification tutorial (@luizcavalcanti, [#1118](https://github.com/RaRe-Technologies/gensim/pull/1118))
* Corpora_and_Vector_Spaces tutorial text clarification (@lgmoneda, [#1116](https://github.com/RaRe-Technologies/gensim/pull/1116))
* Update Transformation and Topics link from quick start notebook (@mariana393, [#1115](https://github.com/RaRe-Technologies/gensim/pull/1115))
* Quick Start Text clarification and typo correction (@luizcavalcanti, [#1114](https://github.com/RaRe-Technologies/gensim/pull/1114))
* Fix typos in Author-topic tutorial (@Fil, [#1102](https://github.com/RaRe-Technologies/gensim/pull/1102))
* Address benchmark inconsistencies in Annoy tutorial (@droudy, [#1113](https://github.com/RaRe-Technologies/gensim/pull/1113))
* Add note about Annoy speed depending on numpy BLAS setup in annoytutorial.ipynb (@greninja, [#1137](https://github.com/RaRe-Technologies/gensim/pull/1137))
* Fix dependencies description on doc2vec-IMDB notebook (@luizcavalcanti, [#1132](https://github.com/RaRe-Technologies/gensim/pull/1132))
* Add documentation for WikiCorpus metadata. (@kirit93, [#1163](https://github.com/RaRe-Technologies/gensim/pull/1163))


## 1.0.0RC2, 2017-02-16

* Add note about Annoy speed depending on numpy BLAS setup in annoytutorial.ipynb (@greninja, [#1137](https://github.com/RaRe-Technologies/gensim/pull/1137))
* Remove direct access to properties moved to KeyedVectors (@tmylk, [#1147](https://github.com/RaRe-Technologies/gensim/pull/1147))
* Remove support for Python 2.6, 3.3 and 3.4 (@tmylk, [#1145](https://github.com/RaRe-Technologies/gensim/pull/1145))
* Write UTF-8 byte strings in tensorboard conversion (@tmylk, [#1144](https://github.com/RaRe-Technologies/gensim/pull/1144))
* Make top_topics and sparse2full compatible with numpy 1.12 strictly int idexing (@tmylk, [#1146](https://github.com/RaRe-Technologies/gensim/pull/1146))

## 1.0.0RC1, 2017-01-31

New features:
* Add Author-topic modeling (@olavurmortensen, [#893](https://github.com/RaRe-Technologies/gensim/pull/893))
* Add FastText word embedding wrapper (@Jayantj, [#847](https://github.com/RaRe-Technologies/gensim/pull/847))
* Add WordRank word embedding  wrapper (@parulsethi, [#1066](https://github.com/RaRe-Technologies/gensim/pull/1066), [#1125](https://github.com/RaRe-Technologies/gensim/pull/1125))
* Add sklearn wrapper for LDAModel (@AadityaJ, [#932](https://github.com/RaRe-Technologies/gensim/pull/932))

Improvements:
* Python 3.6 support (@tmylk [#1077](https://github.com/RaRe-Technologies/gensim/pull/1077))
* Phrases and Phraser allow a generator corpus (ELind77 [#1099](https://github.com/RaRe-Technologies/gensim/pull/1099))
* Ignore DocvecsArray.doctag_syn0norm in save. Fix #789 (@accraze, [#1053](https://github.com/RaRe-Technologies/gensim/pull/1053))
* Move load and save word2vec_format out of word2vec class to KeyedVectors  (@tmylk, [#1107](https://github.com/RaRe-Technologies/gensim/pull/1107))
* Fix bug in LsiModel that occurs when id2word is a Python 3 dictionary. (@cvangysel, [#1103](https://github.com/RaRe-Technologies/gensim/pull/1103)
* Fix broken link to paper in readme (@bhargavvader, [#1101](https://github.com/RaRe-Technologies/gensim/pull/1101))
* Lazy formatting in evaluate_word_pairs (@akutuzov, [#1084](https://github.com/RaRe-Technologies/gensim/pull/1084))
* Deacc option to keywords pre-processing (@bhargavvader, [#1076](https://github.com/RaRe-Technologies/gensim/pull/1076))

Tutorial and doc improvements:

* Clarifying comment in is_corpus func in utils.py (@greninja, [#1109](https://github.com/RaRe-Technologies/gensim/pull/1109))
* Tutorial Topics_and_Transformations fix markdown and add references (@lgmoneda, [#1120](https://github.com/RaRe-Technologies/gensim/pull/1120))
* Fix doc2vec-lee.ipynb results to match previous behavior (@bahbbc, [#1119](https://github.com/RaRe-Technologies/gensim/pull/1119))
* Remove Pattern lib dependency in News Classification tutorial (@luizcavalcanti, [#1118](https://github.com/RaRe-Technologies/gensim/pull/1118))
* Corpora_and_Vector_Spaces tutorial text clarification (@lgmoneda, [#1116](https://github.com/RaRe-Technologies/gensim/pull/1116))
* Update Transformation and Topics link from quick start notebook (@mariana393, [#1115](https://github.com/RaRe-Technologies/gensim/pull/1115))
* Quick Start Text clarification and typo correction (@luizcavalcanti, [#1114](https://github.com/RaRe-Technologies/gensim/pull/1114))
* Fix typos in Author-topic tutorial (@Fil, [#1102](https://github.com/RaRe-Technologies/gensim/pull/1102))
* Address benchmark inconsistencies in Annoy tutorial (@droudy, [#1113](https://github.com/RaRe-Technologies/gensim/pull/1113))


## 0.13.4.1, 2017-01-04

* Disable direct access warnings on save and load of Word2vec/Doc2vec (@tmylk, [#1072](https://github.com/RaRe-Technologies/gensim/pull/1072))
* Making Default hs error explicit (@accraze, [#1054](https://github.com/RaRe-Technologies/gensim/pull/1054))
* Removed unnecessary numpy imports (@bhargavvader, [#1065](https://github.com/RaRe-Technologies/gensim/pull/1065))
* Utils and Matutils changes (@bhargavvader, [#1062](https://github.com/RaRe-Technologies/gensim/pull/1062))
* Tests for the evaluate_word_pairs function (@akutuzov, [#1061](https://github.com/RaRe-Technologies/gensim/pull/1061))

## 0.13.4, 2016-12-22

* Added suggested lda model method and print methods to HDP class (@bhargavvader, [#1055](https://github.com/RaRe-Technologies/gensim/pull/1055))
* New class KeyedVectors to store embedding separate from training code (@anmol01gulati and @droudy, [#980](https://github.com/RaRe-Technologies/gensim/pull/980))
* Evaluation of word2vec models against semantic similarity datasets like SimLex-999 (@akutuzov, [#1047](https://github.com/RaRe-Technologies/gensim/pull/1047))
* TensorBoard word embedding visualisation of Gensim Word2vec format (@loretoparisi, [#1051](https://github.com/RaRe-Technologies/gensim/pull/1051))
* Throw exception if load() is called on instance rather than the class in word2vec and doc2vec (@dust0x, [#889](https://github.com/RaRe-Technologies/gensim/pull/889))
* Loading and Saving LDA Models across Python 2 and 3. Fix #853 (@anmolgulati, [#913](https://github.com/RaRe-Technologies/gensim/pull/913), [#1093](https://github.com/RaRe-Technologies/gensim/pull/1093))
* Fix automatic learning of eta (prior over words) in LDA (@olavurmortensen, [#1024](https://github.com/RaRe-Technologies/gensim/pull/1024)).
    * eta should have dimensionality V (size of vocab) not K (number of topics). eta with shape K x V is still allowed, as the user may want to impose specific prior information to each topic.
    * eta is no longer allowed the "asymmetric" option. Asymmetric priors over words in general are fine (learned or user defined).
    * As a result, the eta update (`update_eta`) was simplified some. It also no longer logs eta when updated, because it is too large for that.
    * Unit tests were updated accordingly. The unit tests expect a different shape than before; some unit tests were redundant after the change; `eta='asymmetric'` now should raise an error.
* Optimise show_topics to only call get_lambda once. Fix #1006. (@bhargavvader, [#1028](https://github.com/RaRe-Technologies/gensim/pull/1028))
* HdpModel doc improvement. Inference and print_topics (@dsquareindia, [#1029](https://github.com/RaRe-Technologies/gensim/pull/1029))
* Removing Doc2Vec defaults so that it won't override Word2Vec defaults. Fix #795. (@markroxor, [#929](https://github.com/RaRe-Technologies/gensim/pull/929))
* Remove warning on gensim import "pattern not installed". Fix #1009 (@shashankg7, [#1018](https://github.com/RaRe-Technologies/gensim/pull/1018))
* Add delete_temporary_training_data() function to word2vec and doc2vec models. (@deepmipt-VladZhukov, [#987](https://github.com/RaRe-Technologies/gensim/pull/987))
* Documentation improvements (@IrinaGoloshchapova, [#1010](https://github.com/RaRe-Technologies/gensim/pull/1010), [#1011](https://github.com/RaRe-Technologies/gensim/pull/1011))
* LDA tutorial by Olavur, tips and tricks (@olavurmortensen, [#779](https://github.com/RaRe-Technologies/gensim/pull/779))
* Add double quote in commmand line to run on Windows (@akarazeev, [#1005](https://github.com/RaRe-Technologies/gensim/pull/1005))
* Fix directory names in notebooks to be OS-independent (@mamamot, [#1004](https://github.com/RaRe-Technologies/gensim/pull/1004))
* Respect clip_start, clip_end in most_similar. Fix #601. (@parulsethi, [#994](https://github.com/RaRe-Technologies/gensim/pull/994))
* Replace Python sigmoid function with scipy in word2vec & doc2vec (@markroxor, [#989](https://github.com/RaRe-Technologies/gensim/pull/989))
* WMD to return 0 instead of inf for sentences that contain a single word (@rbahumi, [#986](https://github.com/RaRe-Technologies/gensim/pull/986))
* Pass all the params through the apply call in lda.get_document_topics(), test case to use the per_word_topics through the corpus in test_ldamodel (@parthoiiitm, [#978](https://github.com/RaRe-Technologies/gensim/pull/978))
* Pyro annotations for lsi_worker (@markroxor, [#968](https://github.com/RaRe-Technologies/gensim/pull/968))


## 0.13.3, 2016-10-20

* Add vocabulary expansion feature to word2vec. (@isohyt, [#900](https://github.com/RaRe-Technologies/gensim/pull/900))
* Tutorial: Reproducing Doc2vec paper result on wikipedia. (@isohyt, [#654](https://github.com/RaRe-Technologies/gensim/pull/654))
* Add Save/Load interface to AnnoyIndexer for index persistence (@fortiema, [#845](https://github.com/RaRe-Technologies/gensim/pull/845))
* Fixed issue [#938](https://github.com/RaRe-Technologies/gensim/issues/938),Creating a unified base class for all topic models. ([@markroxor](https://github.com/markroxor), [#946](https://github.com/RaRe-Technologies/gensim/pull/946))
    -  _breaking change in HdpTopicFormatter.show_\__topics_
* Add Phraser for Phrases optimization. ( @gojomo & @anujkhare , [#837](https://github.com/RaRe-Technologies/gensim/pull/837))
* Fix issue #743, in word2vec's n_similarity method if at least one empty list is passed ZeroDivisionError is raised (@pranay360, [#883](https://github.com/RaRe-Technologies/gensim/pull/883))
* Change export_phrases in Phrases model. Fix issue #794 (@AadityaJ, [#879](https://github.com/RaRe-Technologies/gensim/pull/879))
    - bigram construction can now support multiple bigrams within one sentence
* Fix issue [#838](https://github.com/RaRe-Technologies/gensim/issues/838), RuntimeWarning: overflow encountered in exp ([@markroxor](https://github.com/markroxor), [#895](https://github.com/RaRe-Technologies/gensim/pull/895))
*  Change some log messages to warnings as suggested in issue #828. (@rhnvrm, [#884](https://github.com/RaRe-Technologies/gensim/pull/884))
*  Fix issue #851, In summarizer.py, RunTimeError is raised if single sentence input is provided to avoid ZeroDivionError. (@metalaman, #887)
* Fix issue [#791](https://github.com/RaRe-Technologies/gensim/issues/791), correct logic for iterating over SimilarityABC interface. ([@MridulS](https://github.com/MridulS), [#839](https://github.com/RaRe-Technologies/gensim/pull/839))
* Fix RP model loading for large Fortran-order arrays (@piskvorky, [#605](https://github.com/RaRe-Technologies/gensim/issues/938))
* Remove ShardedCorpus from init because of Theano dependency (@tmylk, [#919](https://github.com/RaRe-Technologies/gensim/pull/919))
* Documentation improvements ( @dsquareindia & @tmylk, [#914](https://github.com/RaRe-Technologies/gensim/pull/914), [#906](https://github.com/RaRe-Technologies/gensim/pull/906) )
* Add Annoy memory-mapping example (@harshul1610, [#899](https://github.com/RaRe-Technologies/gensim/pull/899))
* Fixed issue [#601](https://github.com/RaRe-Technologies/gensim/issues/601), correct docID in most_similar for clip range (@parulsethi, [#994](https://github.com/RaRe-Technologies/gensim/pull/994))

## 0.13.2, 2016-08-19

* wordtopics has changed to word_topics in ldamallet, and fixed issue #764. (@bhargavvader, [#771](https://github.com/RaRe-Technologies/gensim/pull/771))
  - assigning wordtopics value of word_topics to keep backward compatibility, for now
* topics, topn parameters changed to num_topics and num_words in show_topics() and print_topics() (@droudy, [#755](https://github.com/RaRe-Technologies/gensim/pull/755))
  - In hdpmodel and dtmmodel
  - NOT BACKWARDS COMPATIBLE!
* Added random_state parameter to LdaState initializer and check_random_state() (@droudy, [#113](https://github.com/RaRe-Technologies/gensim/pull/113))
* Topic coherence update with `c_uci`, `c_npmi` measures.  LdaMallet, LdaVowpalWabbit support. Add `topics` parameter to coherencemodel. Can now provide tokenized topics to calculate coherence value. Faster backtracking. (@dsquareindia, [#750](https://github.com/RaRe-Technologies/gensim/pull/750), [#793](https://github.com/RaRe-Technologies/gensim/pull/793))
* Added a check for empty (no words) documents before starting to run the DTM wrapper if model = "fixed" is used (DIM model) as this    causes the an error when such documents are reached in training. (@eickho, [#806](https://github.com/RaRe-Technologies/gensim/pull/806))
* New parameters `limit`, `datatype` for load_word2vec_format(); `lockf` for intersect_word2vec_format (@gojomo, [#817](https://github.com/RaRe-Technologies/gensim/pull/817))
* Changed `use_lowercase` option in word2vec accuracy to `case_insensitive` to account for case variations in training vocabulary (@jayantj, [#804](https://github.com/RaRe-Technologies/gensim/pull/804)
* Link to Doc2Vec on airline tweets example in tutorials page (@544895340, [#823](https://github.com/RaRe-Technologies/gensim/pull/823))
* Small error on Doc2vec notebook tutorial (@charlessutton, [#816](https://github.com/RaRe-Technologies/gensim/pull/816))
* Bugfix: Full2sparse clipped to use abs value (@tmylk, [#811](https://github.com/RaRe-Technologies/gensim/pull/811))
* WMD docstring: add tutorial link and query example (@tmylk, [#813](https://github.com/RaRe-Technologies/gensim/pull/813))
* Annoy integration to speed word2vec and doc2vec similarity. Tutorial update (@droudy, [#799](https://github.com/RaRe-Technologies/gensim/pull/799),[#792](https://github.com/RaRe-Technologies/gensim/pull/799) )
* Add converter of LDA model between Mallet, Vowpal Wabit and gensim (@dsquareindia, [#798](https://github.com/RaRe-Technologies/gensim/pull/798), [#766](https://github.com/RaRe-Technologies/gensim/pull/766))
* Distributed LDA in different network segments without broadcast (@menshikh-iv, [#782](https://github.com/RaRe-Technologies/gensim/pull/782))
* Update Corpora_and_Vector_Spaces.ipynb (@megansquire, [#772](https://github.com/RaRe-Technologies/gensim/pull/772))
* DTM wrapper bug fixes caused by renaming num_words in #755 (@bhargavvader, [#770](https://github.com/RaRe-Technologies/gensim/pull/770))
* Add LsiModel.docs_processed attribute (@hobson, [#763](https://github.com/RaRe-Technologies/gensim/pull/763))
* Dynamic Topic Modelling in Python. Google Summer of Code 2016 project. (@bhargavvader, [#739](https://github.com/RaRe-Technologies/gensim/pull/739), [#831](https://github.com/RaRe-Technologies/gensim/pull/831))

## 0.13.1, 2016-06-22

* Topic coherence C_v and U_mass (@dsquareindia, #710)

## 0.13.0, 2016-06-21

* Added Distance Metrics to matutils.pt (@bhargavvader, #656)
* Tutorials migrated from website to ipynb (@j9chan, #721), (@jesford, #733), (@jesford, #725), (@jesford, #716)
* New doc2vec intro tutorial (@seanlaw, #730)
* Gensim Quick Start Tutorial (@andrewjlm, #727)
* Add export_phrases(sentences) to model Phrases (hanabi1224 #588)
* SparseMatrixSimilarity returns a sparse matrix if `maintain_sparsity` is True (@davechallis, #590)
* added functionality for Topics of Words in document - i.e, dynamic topics. (@bhargavvader, #704)
  - also included tutorial which explains new functionalities, and document word-topic colring.
* Made normalization an explicit transformation. Added 'l1' norm support (@dsquareindia, #649)
* added term-topics API for most probable topic for word in vocab. (@bhargavvader, #706)
* build_vocab takes progress_per parameter for smaller output (@zer0n, #624)
* Control whether to use lowercase for computing word2vec accuracy. (@alantian, #607)
* Easy import of GloVe vectors using Gensim (Manas Ranjan Kar, #625)
  - Allow easy port of GloVe vectors into Gensim
  - Standalone script with command line arguments, compatible with Python>=2.6
  - Usage: python -m gensim.scripts.glove2word2vec -i glove_vectors.txt -o output_word2vec_compatible.txt
* Add `similar_by_word()` and `similar_by_vector()` to word2vec (@isohyt, #381)
* Convenience method for similarity of two out of training sentences to doc2vec (@ellolo, #707)
* Dynamic Topic Modelling Tutorial updated with Dynamic Influence Model (@bhargavvader, #689)
* Added function to filter 'n' most frequent words from the dictionary (@abhinavchawla, #718)
* Raise warnings if vocab is single character elements and if alpha is increased in word2vec/doc2vec (@dsquareindia, #705)
* Tests for wikidump (@jonmcoe, #723)
* Mallet wrapper sparse format support (@RishabGoel, #664)
* Doc2vec pre-processing script translated from bash to Python (@andrewjlm, #720)


## 0.12.4, 2016-01-29

* Better internal handling of job batching in word2vec (#535)
  - up to 300% speed up when training on very short documents (~tweets)
* Word2vec CLI in line with original word2vec.c (Andrey Kutuzov, #538)
  - Same default values. See diff https://github.com/akutuzov/gensim/commit/6456cbcd75e6f8720451766ba31cc046b4463ae2
  - Standalone script with command line arguments matching those of original C tool.
  - Usage: python -m gensim.scripts.word2vec_standalone -train data.txt -output trained_vec.txt -size 200 -window 2 -sample 1e-4
* Improved load_word2vec_format() performance (@svenkreiss, #555)
  - Remove `init_sims()` call for performance improvements when normalized vectors are not needed.
  - Remove `norm_only` parameter (API change). Call `init_sims(replace=True)` after the `load_word2vec_format()` call for the old `norm_only=True` behavior.
* Word2vec allows non-strict unicode error handling (ignore or replace) (Gordon Mohr, #466)
* Doc2Vec `model.docvecs[key]` now raises KeyError for unknown keys (Gordon Mohr, #520)
* Fix `DocvecsArray.index_to_doctag` so `most_similar()` returns string doctags (Gordon Mohr, #560)
* On-demand loading of the `pattern` library in utils.lemmatize (Jan Zikes, #461)
  - `utils.HAS_PATTERN` flag moved to `utils.has_pattern()`
* Threadsafe Word2Vec/Doc2Vec finish-check to avoid hang/unending Word2Vec/Doc2Vec training (Gordon Mohr, #571)
* Tuned `TestWord2VecModel.test_cbow_hs()` against random failures (Gordon Mohr, #531)
* Prevent ZeroDivisionError when `default_timer()` indicate no elapsed time (Gordon Mohr, #518)
* Forwards compatibility for NumPy > 1.10 (Matti Lyra, #494, #513)
  - LdaModel and LdaMulticore produce a large number of DeprecationWarnings from
    .inference() because the term ids in each chunk returned from utils.grouper
    are floats. This behaviour has been changed so that the term IDs are now ints.
  - utils.grouper returns a python list instead of a numpy array in .update() when
    LdaModel is called in non distributed mode
  - in distributed mode .update() will still call utils.grouper with as_numpy=True
    to save memory
  - LdaModel.update and LdaMulticore.update have a new keyword parameter
    chunks_as_numpy=True/False (defaults to False) that allows controlling
    this behaviour

## 0.12.3, 2015-11-05

* Make show_topics return value consistent across models (Christopher Corley, #448)
  - All models with the `show_topics` method should return a list of
    `(topic_number, topic)` tuples, where `topic` is a list of
    `(word, probability)` tuples.
  - This is a breaking change that affects users of the `LsiModel`, `LdaModel`,
  and `LdaMulticore` that may be reliant on the old tuple layout of
  `(probability, word)`.
* Mixed integer & string document-tags (keys to doc-vectors) will work (Gordon Mohr, #491)
  - DocvecsArray's `index2doctag` list is renamed/reinterpreted as `offset2doctag`
  - `offset2doctag` entries map to `doctag_syn0` indexes *after* last plain-int doctag (if any)
  - (If using only string doctags, `offset2doctag` may be interpreted same as `index2doctag`.)
* New Tutorials on Dynamic Topic Modelling and Classification via Word2Vec (@arttii #471, @mataddy #500)
* Auto-learning for the eta parameter on the LdaModel (Christopher Corley, #479)
* Python 3.5 support
* Speed improvements to keyword and summarisation methods (@erbas #441)
* OSX wheels (#504)
* Win build (#492)

## 0.12.2, 2015-09-19

* tutorial on text summarization (Ólavur Mortensen, #436)
* more flexible vocabulary construction in word2vec & doc2vec (Philipp Dowling, #434)
* added support for sliced TransformedCorpus objects, so that after applying (for instance) TfidfModel the returned corpus remains randomly indexable. (Matti Lyra, #425)
* changed the LdaModel.save so that a custom `ignore` list can be passed in (Matti Lyra, #331)
* added support for NumPy style fancy indexing to corpus objects (Matti Lyra, #414)
* py3k fix in distributed LSI (spacecowboy, #433)
* Windows fix for setup.py (#428)
* fix compatibility for scipy 0.16.0 (#415)

## 0.12.1, 2015-07-20

* improvements to testing, switch to Travis CI containers
* support for loading old word2vec models (<=0.11.1) in 0.12+ (Gordon Mohr, #405)
* various bug fixes to word2vec, doc2vec (Gordon Mohr, #393, #386, #404)
* TextSummatization support for very short texts (Federico Barrios, #390)
* support for word2vec[['word1', 'word2'...]] convenience API calls (Satish Palaniappan, #395)
* MatrixSimilarity supports indexing generator corpora (single pass)

## 0.12.0, 2015-07-06

* complete API, performance, memory overhaul of doc2vec (Gordon Mohr, #356, #373, #380, #384)
  - fast infer_vector(); optional memory-mapped doc vectors; memory savings with int doc IDs
  - 'dbow_words' for combined DBOW & word skip-gram training; new 'dm_concat' mode
  - multithreading & negative-sampling optimizations (also benefitting word2vec)
  - API NOTE: doc vectors must now be accessed/compared through model's 'docvecs' field
    (eg: "model.docvecs['my_ID']" or "model.docvecs.most_similar('my_ID')")
  - https://github.com/piskvorky/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb
* new "text summarization" module (PR #324: Federico Lopez, Federico Barrios)
  - https://github.com/summanlp/docs/raw/master/articulo/articulo-en.pdf
* new matutils.argsort with partial sort
  - performance speedups to all similarity queries (word2vec, Similarity classes...)
* word2vec can compute likelihood scores for classification (Mat Addy, #358)
  - http://arxiv.org/abs/1504.07295
  - http://nbviewer.ipython.org/github/taddylab/deepir/blob/master/w2v-inversion.ipynb
* word2vec supports "encoding" parameter when loading from C format, for non-utf8 models
* more memory-efficient word2vec training (#385)
* fixes to Python3 compatibility (Pavel Kalaidin #330, S-Eugene #369)
* enhancements to save/load format (Liang Bo Wang #363, Gordon Mohr #356)
  - pickle defaults to protocol=2 for better py3 compatibility
* fixes and improvements to wiki parsing (Lukas Elmer #357, Excellent5 #333)
* fix to phrases scoring (Ikuya Yamada, #353)
* speed up of phrases generation (Dave Challis, #349)
* changes to multipass LDA training (Christopher Corley, #298)
* various doc improvements and fixes (Matti Lyra #331, Hongjoo Lee #334)
* fixes and improvements to LDA (Christopher Corley #323)

## 0.11.0 = 0.11.1 = 0.11.1-1, 2015-04-10

* added "topic ranking" to sort topics by coherence in LdaModel (jtmcmc, #311)
* new fast ShardedCorpus out-of-core corpus (Jan Hajic jr., #284)
* utils.smart_open now uses the smart_open package (#316)
* new wrapper for LDA in Vowpal Wabbit (Dave Challis, #304)
* improvements to the DtmModel wrapper (Yang Han, #272, #277)
* move wrappers for external modeling programs into a submodule (Christopher Corley, #295)
* allow transparent compression of NumPy files in save/load (Christopher Corley, #248)
* save/load methods now accept file handles, in addition to file names (macks22, #292)
* fixes to LdaMulticore on Windows (Feng Mai, #305)
* lots of small fixes & py3k compatibility improvements (Chyi-Kwei Yau, Daniel Nouri, Timothy Emerick, Juarez Bochi, Christopher Corley, Chirag Nagpal, Jan Hajic jr., Flávio Codeço Coelho)
* re-released as 0.11.1 and 0.11.1-1 because of a packaging bug

## 0.10.3, 2014-11-17

* added streamed phrases = collocation detection (Miguel Cabrera, #258)
* added param for multiple word2vec epochs (sebastienj, #243)
* added doc2vec (=paragraph2vec = extension of word2vec) model (Timothy Emerick, #231)
* initialize word2vec deterministically, for increased experiment reproducibility (KCzar, #240)
* all indexed corpora now allow full Python slicing syntax (Christopher Corley, #246)
* update distributed code for new Pyro4 API and py3k (Michael Brooks, Marco Bonzanini, #255, #249)
* fixes to six module version (Lars Buitinck, #259)
* fixes to setup.py (Maxim Avanov and Christopher Corley, #260, #251)
* ...and lots of minor fixes & updates all around

## 0.10.2, 2014-09-18

* new parallelized, LdaMulticore implementation (Jan Zikes, #232)
* Dynamic Topic Models (DTM) wrapper (Arttii, #205)
* word2vec compiled from bundled C file at install time: no more pyximport (#233)
* standardize show_/print_topics in LdaMallet (Benjamin Bray, #223)
* add new word2vec multiplicative objective (3CosMul) of Levy & Goldberg (Gordon Mohr, #224)
* preserve case in MALLET wrapper (mcburton, #222)
* support for matrix-valued topic/word prior eta in LdaModel (mjwillson, #208)
* py3k fix to SparseCorpus (Andreas Madsen, #234)
* fix to LowCorpus when switching dictionaries (Christopher Corley, #237)

## 0.10.1, 2014-07-22

* word2vec: new n_similarity method for comparing two sets of words (François Scharffe, #219)
* make LDA print/show topics parameters consistent with LSI (Bram Vandekerckhove, #201)
* add option for efficient word2vec subsampling (Gordon Mohr, #206)
* fix length calculation for corpora on empty files (Christopher Corley, #209)
* improve file cleanup of unit tests (Christopher Corley)
* more unit tests
* unicode now stored everywhere in gensim internally; accepted input stays either utf8 or unicode
* various fixes to the py3k ported code
* allow any dict-like input in Dictionary.from_corpus (Andreas Madsen)
* error checking improvements to the MALLET wrapper
* ignore non-articles during wiki parsig
* utils.lemmatize now (optionally) ignores stopwords

## 0.10.0 (aka "PY3K port"), 2014-06-04

* full Python 3 support (targeting 3.3+, #196)
* all internal methods now expect & store unicode, instead of utf8
* new optimized word2vec functionality: negative sampling, cbow (sebastien-j, #162)
* allow by-frequency sort in Dictionary.save_as_text (Renaud Richardet, #192)
* add topic printing to HDP model (Tiepes, #190)
* new gensim_addons package = optional install-time Cython compilations (Björn Esser, #197)
* added py3.3 and 3.4 to Travis CI tests
* fix a cbow word2vec bug (Liang-Chi Hsieh)

## 0.9.1, 2014-04-12

* MmCorpus fix for Windows
* LdaMallet support for printing/showing topics
* fix LdaMallet bug when user specified a file prefix (Victor, #184)
* fix LdaMallet output when input is single vector (Suvir)
* added LdaMallet unit tests
* more py3k fixes (Lars Buitinck)
* change order of LDA topic printing (Fayimora Femi-Balogun, #188)

## 0.9.0, 2014-03-16

* save/load automatically single out large arrays + allow mmap
* allow .gz/.bz2 corpus filenames => transparently (de)compressed I/O
* CBOW model for word2vec (Sébastien Jean, #176)
* new API for storing corpus metadata (Joseph Chang, #169)
* new LdaMallet class = train LDA using wrapped Mallet
* new MalletCorpus class for corpora in Mallet format (Christopher Corley, #179)
* better Wikipedia article parsing (Joseph Chang, #170)
* word2vec load_word2vec_format uses less memory (Yves Raimond, #164)
* load/store vocabulary files for word2vec C format (Yves Raimond, #172)
* HDP estimation on new documents (Elliot Kulakow, #153)
* store labels in SvmLight corpus (Ritesh, #152)
* fix word2vec binary load on Windows (Stephanus van Schalkwyk)
* replace numpy.svd with scipy.svd for more stability (Sven Döring, #159)
* parametrize LDA constructor (Christopher Corley, #174)
* steps toward py3k compatibility (Lars Buitinck, #154)

## 0.8.9, 2013-12-26

* use travis-ci for continuous integration
* auto-optimize LDA asymmetric prior (Ben Trahan)
* update for new word2vec binary format (Daren Race)
* doc rendering fix (Dan Foreman-Mackey)
* better LDA perplexity logging
* fix Pyro thread leak in distributed algos (Brian Feeny)
* optimizations in word2vec (Bryan Rink)
* allow compressed input in LineSentence corpus (Eric Moyer)
* upgrade ez_setup, doc improvements, minor fixes etc.

## 0.8.8 (aka "word2vec release"), 2013-11-03

* python3 port by Parikshit Samant: https://github.com/samantp/gensimPy3
* massive optimizations to word2vec (cython, BLAS, multithreading): ~20x-300x speedup
* new word2vec functionality (thx to Ghassen Hamrouni, PR #124)
* new CSV corpus class (thx to Zygmunt Zając)
* corpus serialization checks to prevent overwriting (by Ian Langmore, PR #125)
* add context manager support for older Python<=2.6 for gzip and bz2
* added unittests for word2vec

## 0.8.7, 2013-09-18

* initial version of word2vec, a neural network deep learning algo
* make distributed gensim compatible with the new Pyro
* allow merging dictionaries (by Florent Chandelier)
* new design for the gensim website!
* speed up handling of corner cases when returning top-n most similar
* make Random Projections compatible with new scipy (andrewjOc360, PR #110)
* allow "light" (faster) word lemmatization (by Karsten Jeschkies)
* save/load directly from bzip2 files (by Luis Pedro Coelho, PR #101)
* Blei corpus now tries harder to find its vocabulary file (by Luis Pedro Coelho, PR #100)
* sparse vector elements can now be a list (was: only a 2-tuple)
* simple_preprocess now optionally deaccents letters (ř/š/ú=>r/s/u etc.)
* better serialization of numpy corpora
* print_topics() returns the topics, in addition to printing/logging
* fixes for more robust Windows multiprocessing
* lots of small fixes, data checks and documentation updates

## 0.8.6, 2012-09-15

* added HashDictionary (by Homer Strong)
* support for adding target classes in SVMlight format (by Corrado Monti)
* fixed problems with global lemmatizer object when running in parallel on Windows
* parallelization of Wikipedia processing + added script version that lemmatizes the input documents
* added class method to initialize Dictionary from an existing corpus (by Marko Burjek)

## 0.8.5, 2012-07-22

* improved performance of sharding (similarity queries)
* better Wikipedia parsing (thx to Alejandro Weinstein and Lars Buitinck)
* faster Porter stemmer (thx to Lars Buitinck)
* several minor fixes (in HDP model thx to Greg Ver Steeg)
* improvements to documentation

## 0.8.4, 2012-03-09

* better support for Pandas series input (thx to JT Bates)
* a new corpus format: UCI bag-of-words (thx to Jonathan Esterhazy)
* a new model, non-parametric bayes: HDP (thx to Jonathan Esterhazy; based on Chong Wang's code)
* improved support for new scipy versions (thx to Skipper Seabold)
* lemmatizer support for wikipedia parsing (via the `pattern` python package)
* extended the lemmatizer for multi-core processing, to improve its performance

## 0.8.3, 2011-12-02

* fixed Similarity sharding bug (issue #65, thx to Paul Rudin)
* improved LDA code (clarity & memory footprint)
* optimized efficiency of Similarity sharding

## 0.8.2, 2011-10-31

* improved gensim landing page
* improved accuracy of SVD (Latent Semantic Analysis) (thx to Mark Tygert)
* changed interpretation of LDA topics: github issue #57
* took out similarity server code introduced in 0.8.1 (will become a separate project)
* started using `tox` for testing
* + several smaller fixes and optimizations

## 0.8.1, 2011-10-10

* transactional similarity server: see docs/simserver.html
* website moved from university hosting to radimrehurek.com
* much improved speed of lsi[corpus] transformation:
* accuracy tests of incremental svd: test/svd_error.py and http://groups.google.com/group/gensim/browse_thread/thread/4b605b72f8062770
* further improvements to memory-efficiency of LDA and LSA
* improved wiki preprocessing (thx to Luca de Alfaro)
* model.print_topics() debug fncs now support std output, in addition to logging (thx to Homer Strong)
* several smaller fixes and improvements

## 0.8.0 (Armageddon), 2011-06-28

* changed all variable and function names to comply with PEP8 (numTopics->num_topics): BREAKS BACKWARD COMPATIBILITY!
* added support for similarity querying more documents at once (index[query_documents] in addition to index[query_document]; much faster)
* rewrote Similarity so that it is more efficient and scalable (using disk-based mmap'ed shards)
* simplified directory structure (src/gensim/ is now only gensim/)
* several small fixes and optimizations

## 0.7.8, 2011-03-26

* added `corpora.IndexedCorpus`, a base class for corpus serializers (thx to Dieter Plaetinck). This allows corpus formats that inherit from it (MmCorpus, SvmLightCorpus, BleiCorpus etc.) to retrieve individual documents by their id in O(1), e.g. `corpus[14]` returns document #14.
* merged new code from the LarKC.eu team (`corpora.textcorpus`, `models.logentropy_model`, lots of unit tests etc.)
* fixed a bug in `lda[bow]` transformation (was returning gamma distribution instead of theta). LDA model generation was not affected, only transforming new vectors.
* several small fixes and documentation updates

## 0.7.7, 2011-02-13

* new LDA implementation after Hoffman et al.: Online Learning for Latent Dirichlet Allocation
* distributed LDA
* updated LDA docs (wiki experiments, distributed tutorial)
* matrixmarket header now uses capital 'M's: MatrixMarket. (André Lynum reported than Matlab has trouble processing the lowercase version)
* moved code to github
* started gensim Google group

## 0.7.6, 2011-01-10

* added workaround for a bug in numpy: pickling a fortran-order array (e.g. LSA model) and then loading it back and using it results in segfault (thx to Brian Merrel)
* bundled a new version of ez_setup.py: old failed with Python2.6 when setuptools were missing (thx to Alan Salmoni).

## 0.7.5, 2010-11-03

* further optimization to LSA; this is the version used in my NIPS workshop paper
* got rid of SVDLIBC dependency (one-pass LSA now uses stochastic algo for base-base decompositions)

## 0.7.4

* sped up Latent Dirichlet ~10x (through scipy.weave, optional)
* finally, distributed LDA! scales almost linearly, but no tutorial yet. see the tutorial on distributed LSI, everything's completely analogous.
* several minor fixes and improvements; one nasty bug fixed (lsi[corpus] didn't work; thx to Danilo Spinelli)

## 0.7.3

* added stochastic SVD decomposition (faster than the current one-pass LSI algo, but needs two passes over the input corpus)
* published gensim on mloss.org

## 0.7.2

* added workaround for a numpy bug where SVD sometimes fails to converge for no good reason
* changed content of gensims's PyPi title page
* completed HTML tutorial on distributed LSA

## 0.7.1

* fixed a bug in LSA that occurred when the number of features was smaller than the number of topics (thx to Richard Berendsen)

## 0.7.0

* optimized vocabulary generation in gensim.corpora.dictionary (faster and less memory-intense)
* MmCorpus accepts compressed input (file-like objects such as GzipFile, BZ2File; to save disk space)
* changed sparse solver to SVDLIBC (sparsesvd on PyPi) for large document chunks
* added distributed LSA, updated tutorials (still experimental though)
* several minor bug fixes

## 0.6.0

* added option for online LSI training (yay!). the transformation can now be
  used after any amount of training, and training can be continued at any time
  with more data.
* optimized the tf-idf transformation, so that it is a strictly one-pass algorithm in all cases  (thx to Brian Merrell).
* fixed Windows-specific bug in handling binary files (thx to Sutee Sudprasert)
* fixed 1-based feature counting bug in SVMlight format (thx to Richard Berendsen)
* added 'Topic :: Text Processing :: Linguistic' to gensim's pypi classifiers
* change of sphinx documentation css and layout

## 0.5.0

* finished all tutorials, stable version

## 0.4.7

* tutorial on transformations

## 0.4.6

* added Random Projections (aka Random Indexing), as another transformation model.
* several DML-CZ specific updates

## 0.4.5

* updated documentation
* further memory optimizations in SVD (LSI)

## 0.4.4

* added missing test files to MANIFEST.in

## 0.4.3

* documentation changes
* added gensim reference to Wikipedia articles (SVD, LSI, LDA, TFIDF, ...)

## 0.4.2

* finally, a tutorial!
* similarity queries got their own package

## 0.4.1

* pdf documentation
* removed dependency on python2.5 (theoretically, gensim now runs on 2.6 and 2.7 as well).

## 0.4.0

* support for ``python setup.py test``
* fixing package metadata
* documentation clean-up

## 0.2.0

* First version
