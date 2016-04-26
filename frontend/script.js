$(function (){

    var $form = $("form"),
        $text = $form.find('[name="text"]'),
        $results = $("#results"),
        endpoint = "YOUR_API_GATEWAY_ENDPOINT_HERE";

    $form.submit(function (e){

        e.preventDefault();

        var text = $text.val();

        $.getJSON(endpoint, {
            'text': text
        }).success(function (json){

            var sentiment = json.sentiment;

            $results.text("The predicted sentiment is " + sentiment + ".");

        }).fail(function (xhr){

            window.console && console.error(xhr);
            $results.text("Something went wrong!");

        });

    });

});