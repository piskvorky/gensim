# How to submit an issue?

First, please see [contribution-guide.org](http://www.contribution-guide.org/) for the steps we expect from contributors before submitting an issue or bug report. Be as concrete as possible, include relevant logs, package versions etc.

Also, please check the [Gensim FAQ](https://github.com/RaRe-Technologies/gensim/wiki/Recipes-&-FAQ) page before posting.

**The proper place for open-ended questions is the [Gensim mailing list](https://groups.google.com/forum/#!forum/gensim).** Github is not the right place for research discussions or feature requests.

# How to add a new feature or create a pull request?

1. <a href="https://github.com/RaRe-Technologies/gensim/fork">Fork the Gensim repository</a>
2. Clone your fork: `git clone https://github.com/<YOUR_GITHUB_USERNAME>/gensim.git`
3. Create a new branch based on `develop`: `git checkout -b my-feature develop`
4. Setup your Python enviroment
   - Create a new [virtual environment](https://virtualenv.pypa.io/en/stable/): `pip install virtualenv; virtualenv gensim_env;` for linux: `source gensim_env/bin/activate;` for windows: `gensim_env\Scripts\activate;`
   - Install Gensim and its test dependencies in [editable mode](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs): 
      - For linux: `pip install -e .[test]`
      - For windows: `pip install -e .[test-win]`
5. Implement your changes
6. Check that everything's OK in your branch:
   - Check it for PEP8: `tox -e flake8`
   - Build its documentation (works only for MacOS/Linux): `tox -e docs` (documentation stored in `docs/src/_build`)
   - Run unit tests: `tox -e py{version}-{os}`, for example `tox -e py27-linux` or `tox -e py36-win` where
      - `{version}` is one of `27`, `35`, `36`
      - `{os}` is either `win` or `linux`
7. Add files, commit and push: `git add ... ; git commit -m "my commit message"; git push origin my-feature`
8. [Create a PR](https://help.github.com/articles/creating-a-pull-request/) on Github. Write a **clear description** for your PR, including all the context and relevant information, such as:
   - The issue that you fixed, e.g. `Fixes #123`
   - Motivation: why did you create this PR? What functionality did you set out to improve? What was the problem + an overview of how you fixed it? Whom does it affect and how should people use it?
   - Any other useful information: links to other related Github or mailing list issues and discussions, benchmark graphs, academic papers…

P.S. for developers: see our [Developer Page](https://github.com/piskvorky/gensim/wiki/Developer-page#code-style) for details on the Gensim code style, CI, testing and similar.

**Thanks and let's improve the open source world together!**
