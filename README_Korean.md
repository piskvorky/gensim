gensim – Python의 주제 모델링
==================================

<!--
The following image URLs are obfuscated = proxied and cached through
Google because of Github's proxying issues. See:
https://github.com/RaRe-Technologies/gensim/issues/2805
-->

[![Build Status](https://github.com/RaRe-Technologies/gensim/actions/workflows/tests.yml/badge.svg?branch=develop)](https://github.com/RaRe-Technologies/gensim/actions)
[![GitHub release](https://img.shields.io/github/release/rare-technologies/gensim.svg?maxAge=3600)](https://github.com/RaRe-Technologies/gensim/releases)
[![Downloads](https://img.shields.io/pypi/dm/gensim?color=blue)](https://pepy.tech/project/gensim/)
[![DOI](https://zenodo.org/badge/DOI/10.13140/2.1.2393.1847.svg)](https://doi.org/10.13140/2.1.2393.1847)
[![Mailing List](https://img.shields.io/badge/-Mailing%20List-blue.svg)](https://groups.google.com/forum/#!forum/gensim)
[![Follow](https://img.shields.io/twitter/follow/gensim_py.svg?style=social&style=flat&logo=twitter&label=Follow&color=blue)](https://twitter.com/gensim_py)

Gensim은 주제 모델링 , 문서 인덱싱 및 대규모 말뭉치의 유사성 검색 을 위한 Python 라이브러리입니다 . 대상은 자연어 처리 (NLP) 및 정보 검색 (IR) 커뮤니티입니다.

## ⚠️  이 오픈 소스 프로젝트를 지속할 수 있도록 Gensim 을 후원 해 주십시오. ❤️


특징
--------

-   모든 알고리즘은 코퍼스 크기(RAM, 스트리밍, 코어 외)보다 큰 입력을 처리할 수 있는 메모리 독립적입니다 .
-   직관적인 인터페이스
    -   자신의 입력 말뭉치/데이터스트림(사소한 스트리밍 API)을 쉽게 연결할 수 있습니다.
    -   다른 벡터 공간 알고리즘으로 쉽게 확장 가능(사소한 변환 API)
-   온라인 잠재 의미 분석(LSA/LSI/SVD) , 잠재 디리클레 할당(LDA) , 랜덤 투영(RP) , 계층적 디리클레 프로세스(HDP) 또는 word2vec 딥 러닝 과 같은 인기 있는 알고리즘의 효율적인 멀티코어 구현 .
-   분산 컴퓨팅 : 컴퓨터 클러스터에서 잠재 의미 분석 및 잠재 디리클레 할당 을 실행할 수 있습니다.
-   광범위한 문서 및 Jupyter Notebook 자습서.

이 기능 목록이 머리를 긁적였다면 먼저 Wikipedia 에서 Vector Space Model 및 unsupervised document analysis 에 대해 자세히 읽을 수 있습니다 .

설치
------------

이 소프트웨어는 과학 컴퓨팅을 위한 두 개의 Python 패키지인 NumPy 및 Scipy 에 의존합니다 . gensim을 설치하기 전에 설치해야 합니다.

또한 NumPy를 설치하기 전에 빠른 BLAS 라이브러리를 설치하는 것이 좋습니다. 이것은 선택 사항이지만 MKL, ATLAS 또는 OpenBLAS 와 같은 최적화된 BLAS를 사용하면 성능이 몇 배나 향상되는 것으로 알려져 있습니다. OSX에서 NumPy는 vecLib BLAS를 자동으로 선택하므로 특별한 작업을 수행할 필요가 없습니다.

최신 버전의 gensim 설치:

```bash
    pip install --upgrade gensim
```

또는 대신 소스 tar.gz 패키지 를 다운로드하고 압축을 푼 경우 :

```bash
    python setup.py install
```

다른 설치 모드는 설명서 를 참조하십시오 .

Gensim은 지원되는 모든 Python 버전 에서 지속적으로 테스트 되고 있습니다 . Python 2.7에 대한 지원은 gensim 4.0.0에서 삭제되었습니다. Python 2.7을 사용해야 하는 경우 gensim 3.8.3을 설치하십시오.

gensim이 왜 그렇게 빠르고 메모리 효율적입니까? 순수 파이썬이 아니고 파이썬이 느리고 욕심이 많은 것 아닙니까?
--------------------------------------------------------------------------------------------------------

많은 과학적 알고리즘은 큰 행렬 연산으로 표현될 수 있습니다(위의 BLAS 참고 사항 참조). Gensim은 NumPy에 대한 종속성을 통해 이러한 저수준 BLAS 라이브러리를 활용합니다. 따라서 gensim-the-top-level-code는 순수한 Python이지만 실제로는 다중 스레딩을 포함하여 후드 아래에서 고도로 최적화된 Fortran/C를 실행합니다(BLAS가 그렇게 구성된 경우).

메모리 측면에서 gensim은 스트리밍 데이터 처리를 위해 Python의 내장 생성기와 반복기를 많이 사용합니다. 메모리 효율성은 gensim의 설계 목표 중 하나였으며 gensim의 핵심 기능입니다.

문서
-------------

-   빠른 시작
-   튜토리얼
-   공식 API 문서

  [QuickStart]: https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html
  [Tutorials]: https://radimrehurek.com/gensim/auto_examples/
  [Official Documentation and Walkthrough]: http://radimrehurek.com/gensim/
  [Official API Documentation]: http://radimrehurek.com/gensim/apiref.html

지원
-------
상업적 지원은 Gensim 후원을 참조하십시오 .

공개 Gensim 메일링 리스트 에서 개방형 질문을 하십시오 .

Github에서 버그를 제기 하지만 문제 템플릿 을 따라야 합니다 . 버그가 아니거나 요청된 내용을 제공하지 못한 문제는 점검 없이 종료됩니다.




---------

채택자
--------

| Company | Logo | Industry | Use of Gensim |
|---------|------|----------|---------------|
| [RARE Technologies](http://rare-technologies.com) | ![rare](docs/src/readme_images/rare.png) | ML & NLP consulting | Creators of Gensim – this is us! |
| [Amazon](http://www.amazon.com/) |  ![amazon](docs/src/readme_images/amazon.png) | Retail |  Document similarity. |
| [National Institutes of Health](https://github.com/NIHOPA/pipeline_word2vec) | ![nih](docs/src/readme_images/nih.png) | Health | Processing grants and publications with word2vec. |
| [Cisco Security](http://www.cisco.com/c/en/us/products/security/index.html) | ![cisco](docs/src/readme_images/cisco.png) | Security |  Large-scale fraud detection. |
| [Mindseye](http://www.mindseyesolutions.com/) | ![mindseye](docs/src/readme_images/mindseye.png) | Legal | Similarities in legal documents. |
| [Channel 4](http://www.channel4.com/) | ![channel4](docs/src/readme_images/channel4.png) | Media | Recommendation engine. |
| [Talentpair](http://talentpair.com) | ![talent-pair](docs/src/readme_images/talent-pair.png) | HR | Candidate matching in high-touch recruiting. |
| [Juju](http://www.juju.com/)  | ![juju](docs/src/readme_images/juju.png) | HR | Provide non-obvious related job suggestions. |
| [Tailwind](https://www.tailwindapp.com/) | ![tailwind](docs/src/readme_images/tailwind.png) | Media | Post interesting and relevant content to Pinterest. |
| [Issuu](https://issuu.com/) | ![issuu](docs/src/readme_images/issuu.png) | Media | Gensim's LDA module lies at the very core of the analysis we perform on each uploaded publication to figure out what it's all about. |
| [Search Metrics](http://www.searchmetrics.com/) | ![search-metrics](docs/src/readme_images/search-metrics.png) | Content Marketing | Gensim word2vec used for entity disambiguation in Search Engine Optimisation. |
| [12K Research](https://12k.co/) | ![12k](docs/src/readme_images/12k.png)| Media |   Document similarity analysis on media articles. |
| [Stillwater Supercomputing](http://www.stillwater-sc.com/) | ![stillwater](docs/src/readme_images/stillwater.png) | Hardware | Document comprehension and association with word2vec. |
| [SiteGround](https://www.siteground.com/) |  ![siteground](docs/src/readme_images/siteground.png) | Web hosting | An ensemble search engine which uses different embeddings models and similarities, including word2vec, WMD, and LDA. |
| [Capital One](https://www.capitalone.com/) | ![capitalone](docs/src/readme_images/capitalone.png) | Finance | Topic modeling for customer complaints exploration. |

-------

gensim 인용
------------

학술 논문 및 논문에서 gensim을 인용 할 때 다음 BibTeX 항목을 사용하십시오.

    @inproceedings{rehurek_lrec,
          title = {{Software Framework for Topic Modelling with Large Corpora}},
          author = {Radim {\v R}eh{\r u}{\v r}ek and Petr Sojka},
          booktitle = {{Proceedings of the LREC 2010 Workshop on New
               Challenges for NLP Frameworks}},
          pages = {45--50},
          year = 2010,
          month = May,
          day = 22,
          publisher = {ELRA},
          address = {Valletta, Malta},
          note={\url{http://is.muni.cz/publication/884893/en}},
          language={English}
    }

  [citing gensim in academic papers and theses]: https://scholar.google.com/citations?view_op=view_citation&hl=en&user=9vG_kV0AAAAJ&citation_for_view=9vG_kV0AAAAJ:NaGl4SEjCO4C

  [design goals]: http://radimrehurek.com/gensim/about.html
  [RaRe Technologies]: http://rare-technologies.com/wp-content/uploads/2016/02/rare_image_only.png%20=10x20
  [rare\_tech]: //rare-technologies.com
  [Talentpair]: https://avatars3.githubusercontent.com/u/8418395?v=3&s=100
  [citing gensim in academic papers and theses]: https://scholar.google.cz/citations?view_op=view_citation&hl=en&user=9vG_kV0AAAAJ&citation_for_view=9vG_kV0AAAAJ:u-x6o8ySG0sC

  [documentation and Jupyter Notebook tutorials]: https://github.com/RaRe-Technologies/gensim/#documentation
  [Vector Space Model]: http://en.wikipedia.org/wiki/Vector_space_model
  [unsupervised document analysis]: http://en.wikipedia.org/wiki/Latent_semantic_indexing
  [NumPy and Scipy]: http://www.scipy.org/Download
  [ATLAS]: http://math-atlas.sourceforge.net/
  [OpenBLAS]: http://xianyi.github.io/OpenBLAS/
  [source tar.gz]: http://pypi.python.org/pypi/gensim
  [documentation]: http://radimrehurek.com/gensim/install.html