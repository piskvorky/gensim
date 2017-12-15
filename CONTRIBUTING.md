**How to submit issue?**
------------------------
First, please see [contribution-guide.org](http://www.contribution-guide.org/) for steps we expect from contributors before submitting an issue or bug report. Be as concrete as possible, include relevant logs, package versions etc.

Also, please check the [Gensim FAQ](https://github.com/RaRe-Technologies/gensim/wiki/Recipes-&-FAQ) page before posting.

**The proper place for open-ended questions is the [gensim mailing list](https://groups.google.com/forum/#!forum/gensim).** Github is not the right place for research discussions or feature requests.

**How to add new feature / create PR / etc?**
---------------------

1. <a href="https://github.com/RaRe-Technologies/gensim/fork">Fork gensim repository</a>
2. Clone your fork: `git clone https://github.com/<USERNAME>/gensim.git`
3. Create new branch based on develop: `git checkout -b my-feature develop`
4. Make all needed changes
4. Check that all OK in your branch (in needed):
   - Check PEP8: `tox -e flake8`
   - Build documentation (work only for MacOS/Linux): `tox -e docs`
   - Run base tests: `tox -e py{version}-{os}`, for example `tox -e py27-linux` or `tox -e py36-win` where
      - `{version}` from `27`, `35`, `36` and 
      - `{os}` from `win`, `linux`
      
5. Add files, commit and push: `git add ... ; git commit -m "my commit message"; git push origin my-feature`
6. Create PR on github. Please add clear description for PR and add all needed information to first message, for example:
   - Number of issue that you fixed, like `#123`
   - Motivation (why and how)
   - Any useful related information
   - ...


P/S: for developers: see our [Developer Page](https://github.com/piskvorky/gensim/wiki/Developer-page#code-style) for details on code style, testing and similar.

Thanks!
