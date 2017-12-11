// modded from https://github.com/TylerLH/github-latest-commits-widget

githubCommits = function() {
  var username = 'RaRe-Technologies';
  var repo = 'gensim';
  var limit = '3';

  var callback = function(response) {
    var items = response.data;
    var ul = $('#commit-history-widget');
    ul.empty();

    var _results = [];
    for (var index in items) {
      result = items[index];
      _results.push((function(index, result) {
        if (result.author != null) {
          var widget = "<li>\n" +
                         "<div class=\"left\">\n" +
                           "<img class=\"commit-avatar\" src=\"" + result.author.avatar_url + "\">\n" +
                         "</div>\n" +
                         "<div class=\"commit-author-info left\">\n" +
                           "<a href=\"https://github.com/" + result.author.login + "\">" +
                             "<b class=\"commit-author\">" + result.author.login + "</b>" +
                           "</a>\n" +
                           "<br />\n" +
                           "<b class=\"commit-date\">" + ($.timeago(result.commit.committer.date)) + "</b>" +
                           "<br />" +
                           "<i class=\"commit-sha\">SHA: " + result.sha + "</i>\n" +
                           "<br />\n" +
                           "<a class=\"commit-message\"" +
                             "href=\"https://github.com/" + username + "/" + repo + "/commit/" + result.sha + "\""+
                             "target=\"_blank\">" + result.commit.message +
                           "</a>\n" +
                         "</div>\n" +
                       "</li>"
          return ul.append(widget);
        }
      })(index, result));
    }
    return _results;
  };

  url = "https://api.github.com/repos/" + username + "/" + repo + "/commits?callback=callback";
  return $.ajax(url, {
    data: {
      per_page: limit
    },
    dataType: "jsonp",
    type: "get",
    success: function(response) {
      return callback(response);
    }
  });
};
