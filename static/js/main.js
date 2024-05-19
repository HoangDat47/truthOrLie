$(document).ready(function () {
    // Init
    $('.video-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                var video = document.getElementById('videoPreview');
                video.src = URL.createObjectURL(input.files[0]);
                video.style.display = 'block';
            }
            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#videoUpload").change(function () {
        $('.video-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').html(`
                    <div style="margin:10px;">Lie: <div class="progress">
                        <div class="progress-bar progress-bar-striped" role="progressbar" style="width: ${data.lie}%" aria-valuenow="${data.lie}" aria-valuemin="0" aria-valuemax="100">${data.lie.toFixed(2)}%</div>
                    </div></div>
                    <div style="margin:10px;">Truth: <div class="progress">
                        <div class="progress-bar progress-bar-striped" role="progressbar" style="width: ${data.truth}%" aria-valuenow="${data.truth}" aria-valuemin="0" aria-valuemax="100">${data.truth.toFixed(2)}%</div>
                    </div></div>
                `);
                console.log('Success!');
            },
        });
    });
});