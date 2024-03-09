var _fileData;
var _file_count = 0;
var _file_array = new Array();
var _filename_array = new Array();

const PreviewImg = document.getElementById("transcript");

$(document).ready(function(){

    // init UI
    var isStart = false;
    _file_array.length = 0;
    _filename_array.length = 0;

    var textElem = document.getElementById("Add-Text");
    var textSelected = null;
    textElem.onclick = function(e) {
        textSelected = this.value;
        this.value = null;
    }

    textElem.value = null;
    textElem.onchange = function(e) { // will trigger each time
        if (!this.files) return;

        _file_count = this.files.length;
        _file_array.length = 0;
        _filename_array.length = 0;
        for(var i=0; i<_file_count; i++) {
            var reader = new FileReader();
            _filename_array.push(this.files[i].name)
            reader.onload = function (e) {
                _fileData = e.target.result;
                _file_array.push(_fileData);
            };
            reader.readAsDataURL(this.files[i]);
        }
    };

    function handleFileDialog(changed) {
        // boolean parameter if the value has changed (new select) or not (canceled)

    }

    $(".submit-btn").click(function (e) {
        $("#loading").show();

        // _fileData = _file_array.pop()

        if (_file_count == 0) {
            alert("please select json text file.");
            return;
        }

        var jsonObj = {
            'file': _file_array,
            'names': _filename_array
        }

        $.ajax({
            url: "/get-malware_predict",
            type: "POST",
            contentType:"application/json; charset=utf-8",
            data: JSON.stringify(jsonObj),
            dataType: "json",
            success: function (response) {
                isStart = true;
                var status = response['status'];
                var result = response['message'];

                $(".result").hide();
                ttext = result;
                html='<ul>';
                html += ('<span data-m=' + '>' + ttext + ' </span>');
                html += '</ul>';
                $("#click-text").html(html);
                $("#click-text").show();
                $("#loading").hide();
            },
            error: function (request, response) {
                isStart = false;
                alert("Web server Error. Try again later.");
                return ;
            },
            complete: function(response) {
            }
        });
    });

});


function clickEvent(index) {

}

function play_slice(tm) {

}

