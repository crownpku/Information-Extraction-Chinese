function logsumexp(x) {
    var a = Math.max.apply(null, x);
    var total = 0.0;
    for (var i = 0; i < x.length; i++) {
        total += Math.exp(x[i] - a);
    }
    return Math.log(total) + a;
}

function set_mention_callbacks(hover_callbacks, unhover_callbacks, head_scores, mention_start, mention_end) {
    var scores = [];
    for (var token = mention_start; token <= mention_end; token++) {
        scores.push(head_scores[token]);
    }
    var log_norm = logsumexp(scores);

    hover_callbacks.add(function() {
        var show_attention = $("#show-attention").is(":checked");
        for (var token = mention_start; token <= mention_end; token++) {
            var token_span = $("#t" + token);
            token_span.css("font-weight", "normal");
            if (show_attention) {
                var p = Math.exp(head_scores[token] - log_norm);
                var r = Math.round(-36 * p + 255);
                var g = Math.round(-187 * p + 255);
                var b = Math.round(-200 * p + 255);
                var c = p > 0.75 ? "lightgray" : "black";
                token_span
                    .css("color", c)
                    .css("background-color", "rgb(" + r + "," + g + "," + b + ")");
            }
        }
        $("#lrb" + mention_start).text("(");
        $("#rrb" + mention_end).text(")");
    });
    unhover_callbacks.add(function() {
        for (var token = mention_start; token <= mention_end; token++) {
            $("#t" + token)
                .css("color", "black")
                .css("font-weight", "300")
                .css("background-color", "white");
        }
        $("#lrb" + mention_start).text("");
        $("#rrb" + mention_end).text("");
    });
}

function render_predicted_clusters(clusters_div, clusters_data, text, head_scores) {
    clusters_div.empty();
    $.each(clusters_data, function(i, cluster) {
        var item = $("<li>")
                .addClass("list-group-item")
                .appendTo(clusters_div);

        var hover_callbacks = $.Callbacks();
        var unhover_callbacks = $.Callbacks();
        $.each(cluster, function(j, mention) {
            if (j != 0) {
                $("<span>").html(", ").appendTo(item);
            }
            var mention_start = mention[0];
            var mention_end = mention[1];
            set_mention_callbacks(hover_callbacks, unhover_callbacks, head_scores, mention_start, mention_end);
            var mention_text = text.slice(mention_start, mention_end + 1).join(" ");
            $("<span>")
                .html(mention_text)
                .appendTo(item);
        });

        item.hover(hover_callbacks.fire, unhover_callbacks.fire);
    });
}

function render_top_spans(top_spans_div, top_spans_data, text, head_scores) {
    top_spans_div.empty();
    $.each(top_spans_data, function(i, mention) {
        var item = $("<li>")
                .addClass("list-group-item")
                .appendTo(top_spans_div);

        var hover_callbacks = $.Callbacks();
        var unhover_callbacks = $.Callbacks();
        var mention_start = mention[0];
        var mention_end = mention[1];
        set_mention_callbacks(hover_callbacks, unhover_callbacks, head_scores, mention_start, mention_end);
        var mention_text = text.slice(mention_start, mention_end + 1).join(" ");
        $("<span>")
            .html(mention_text)
            .appendTo(item);
        item.hover(hover_callbacks.fire, unhover_callbacks.fire);
    });
}

function jsonCallback(data) {
    var text_div = $("#text");
    text_div.empty();
    var text = [].concat.apply([], data.sentences);
    $.each(text, function(i, token) {
        var lrb = $("<span>")
                .attr("id", "lrb" + i)
                .appendTo(text_div);
        var text_span =  $("<span>")
                .attr("id", "t" + i)
                .html(token)
                .appendTo(text_div);
        var rrb = $("<span>")
                .attr("id", "rrb" + i)
                .appendTo(text_div);
        $("<span>").text(" ").appendTo(text_div);
    });
    render_predicted_clusters($("#predicted-clusters"), data.predicted_clusters, text, data.head_scores);
    render_top_spans($("#top-spans"), data.top_spans, text, data.head_scores);
    $("#top-spans-title").text("Top Spans");
    $("#text-title").text("Document");
    $("#predicted-clusters-title").text("Predicted Clusters");
}

$(document).ready(function() {
    $("#input-form").submit(function(event){
        event.preventDefault();
        var text = $("#input-doc").val().replace("’", "'");
        $.getJSON("https://appositive.cs.washington.edu:8080?callback=?", {"text": text});
    });
});
