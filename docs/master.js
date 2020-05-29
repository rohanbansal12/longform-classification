$(document).ready(function() {

  $('a').click(function() {
    $(this).attr('target', '_blank');
});
  $("button").click(function() {
    $("table").remove();
    $("#spinner").show();
    var a = $("#s1").val();
    var b = $("#s2").val();
    var c = $("#s3").val();
    var d = $("#s4").val();
    var e = $("#s5").val();
    const Url = "https://jp0pvkfc6f.execute-api.us-east-2.amazonaws.com/default/rank_news_demo";
    var Data = {
      change: 1,
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


})()
});
});
