# How to submit an issue?

First, please see [contribution-guide.org](http://www.contribution-guide.org/) for the steps we expect from contributors before submitting an issue or bug report. Be as concrete as possible, include relevant logs, package versions etc.

Also, please check the [Gensim FAQ](https://github.com/RaRe-Technologies/gensim/wiki/Recipes-&-FAQ) page before posting.

**The proper place for open-ended questions is the [Gensim mailing list](https://groups.google.com/g/gensim).** Github is not the right place for research discussions or feature requests.

# How to add a new feature or create a pull request?

1. <a href="https://github.com/RaRe-Technologies/gensim/fork">Fork the Gensim repository</a>
2. Clone your fork: `git clone https://github.com/<YOUR_GITHUB_USERNAME>/gensim.git`
3. Create a new branch based on `develop`: `git checkout -b my-feature develop`
4. Setup your Python enviroment
   - Create a new [virtual environment](https://virtualenv.pypa.io/en/stable/): `pip install virtualenv; virtualenv gensim_env` and activate it:
      - For linux: `source gensim_env/bin/activate` 
      - For windows: `gensim_env\Scripts\activate`
   - Install Gensim and its test dependencies in [editable mode](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs): 
      - For linux: `pip install -e .[test]`
      - For windows: `pip install -e .[test-win]`
5. Implement your changes
6. Check that everything's OK in your branch:
   - Check it for PEP8: `flake8 --ignore E12,W503 --max-line-length 120 --show-source gensim`
   - Build its documentation (works only for MacOS/Linux): `make -C docs/src html` (documentation stored in `docs/src/_build`)
   - Run unit tests: `pytest -v gensim/test`
7. Add files, commit and push: `git add ... ; git commit -m "my commit message"; git push origin my-feature`
8. [Create a PR](https://help.github.com/articles/creating-a-pull-request/) on Github. Write a **clear description** for your PR, including all the context and relevant information, such as:
   - The issue that you fixed, e.g. `Fixes #123`
   - Motivation: why did you create this PR? What functionality did you set out to improve? What was the problem + an overview of how you fixed it? Whom does it affect and how should people use it?
   - Any other useful information: links to other related Github or mailing list issues and discussions, benchmark graphs, academic papers…

P.S. for developers: see our [Developer Page](https://github.com/piskvorky/gensim/wiki/Developer-page#code-style) for details on the Gensim code style, CI, testing and similar.

**Thanks and let's improve the open source world together!**
