$(document).ready(function() {

  $('a').click(function() {
    $(this).attr('target', '_blank');
});
  $("button").click(function() {
    $("#rank_results").remove();
    var current_selected_tab = $("div.mdl-layout__tab-bar a.is-active").text();
    $("#spinner").show();
    var a = $("#s1").val();
    var b = $("#s2").val();
    var c = $("#s3").val();
    var d = $("#s4").val();
    var e = $("#s5").val();
    const Url = "https://jp0pvkfc6f.execute-api.us-east-2.amazonaws.com/default/rank_news_demo";
    var Data = {
      dataset: 1,
      a: a,
      b: b,
      c: c,
      d: d,
      e: e
    };
    console.log(Data);
    var payLoad = {
      method: "POST",
      mode: "cors",
      body: JSON.stringify(Data)
    };

(async () => {
  let ranked_articles = await fetch(Url, payLoad)
  .then(response => response.json())
  .then(data => {return data});
  var grand_html = ""
  ranked_articles.forEach(function(article) {
    grand_html += "<tr>";
    grand_html += '<td class="title mdl-data-table__cell--non-numeric"><a class="article_link" href="';
    grand_html += article['link'];
    grand_html += '">';
    grand_html += article['title'];
    grand_html += '<br></br><br></br>';
    grand_html += '<p class="top_words">';
    article['top_words'].forEach(function(word) {
      if (word.length > 2 && isNaN(word)) {
        grand_html += word;
        grand_html += ",";
      };
    });
    grand_html = grand_html.slice(0, -1)
    grand_html += '</p>';
    grand_html += '<p class="least_words">';
    article['least_words'].forEach(function(word) {
      if (word.length > 2 && isNaN(word)) {
        grand_html += word;
        grand_html += ",";
      };
    });
    grand_html = grand_html.slice(0, -1)
    grand_html += '</p>';
    grand_html += '</td>';
    grand_html += '<td class="logit">';
    grand_html += article['logit'].toString();
    grand_html += '</td>'
    grand_html += '</tr>'
  });
  grand_html += '</tbody>'
  grand_html += '</table>'
  var prepend = '<table id="rank_results" style="width:90%; margin-left:5%; margin-right:5%;" class="mdl-data-table mdl-js-data-table">'
  prepend += "<thead>"
  prepend += '<tr><th class="mdl-data-table__cell--non-numeric">Articles</th><th class="mdl-data-table__cell--non-numeric">Prediction</th></tr></thead><tbody>'
  var final_html_str = prepend + grand_html
  $("#spinner").hide();
  $("#table-wrapper-browser").append(final_html_str)
})()
});
});
