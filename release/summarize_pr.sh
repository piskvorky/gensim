#
# Summarize pull requests.  Example usage:
#
#   bash summarize_pr.sh 123 456 789
#
# where 123, 456 and 789 are PR IDs.  List as many PRs as you want.
#
# Gives a summary like:
#
# - Removes native xz support (PR [#282](https://github.com/RaRe-Technologies/smart_open/pull/282), [@tdhopper](https://github.com/tdhopper))
# - backward compatibility fixes (PR [#294](https://github.com/RaRe-Technologies/smart_open/pull/294), [@mpenkov](https://github.com/mpenkov))
#
# You can paste this directly into CHANGELOG.md
#
for prid in "$@"
do
    api_url="https://api.github.com/repos/RaRe-Technologies/gensim/pulls/${prid})"
    json="$(wget --quiet -O - ${api_url})"
    title="$(echo "$json" | jq .title --raw-output)"
    html_url="$(echo "$json" | jq .html_url --raw-output)"
    user="$(echo "$json" | jq .user.login --raw-output)"
    user_html_url="$(echo "$json" | jq .user.html_url --raw-output)"
    echo "* ${title} (PR [#${prid}](${html_url}), __[@${user}](${user_html_url})__)"
done
