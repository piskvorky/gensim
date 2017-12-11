$(function() {

    var mdTags = $("div").find("[data-type='md']");

    if(mdTags.length > 0) {
        var converter = new showdown.Converter();
        $.each(mdTags, function (i, mdTag) {
            var $mdTag = $(mdTag);

            var mdUrl = $mdTag.data('url');

            $.ajax({
                url: mdUrl,
                dataType: "text",
                success: function (data) {
                    var $container = $mdTag.find('.md-container');
                    var html = converter.makeHtml(data);
                    $container.html(html);
                }
            });
        });
    }
});

