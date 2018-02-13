function drawImage() {
    var canvas_list = document.querySelectorAll("canvas");
    canvas_list.forEach(
        function (curr_canvas, curr_idx, listObj) {
            _draw_image(curr_canvas, 0, 0, 1.0);
            curr_canvas.addEventListener('mouseout',  function(e){_draw_image(curr_canvas, 0, 0, 1.0);}, false);
            curr_canvas.addEventListener('mousemove', function(e){_move(curr_canvas, e);}, false);
        }
    );
}
function _move(curr_canvas, event){
    var rect = curr_canvas.getBoundingClientRect();
    var x_offset = (event.clientX- rect.left) / curr_canvas.width;
    var y_offset = (event.clientY- rect.top) / curr_canvas.height;
    _draw_image(curr_canvas, x_offset, y_offset, 2.0);
}

function _draw_image(curr_canvas, x_offset, y_offset, zoom_level) {

    var ctx = curr_canvas.getContext("2d");
    var detections = JSON.parse(curr_canvas.getAttribute('detections')).bboxes;
    var img_title = curr_canvas.getAttribute('src').split('/').pop();
    var img = new Image();

    img.onload = function () {
        ctx.clearRect(0, 0, curr_canvas.width, curr_canvas.height);
        var x_top =  x_offset*(img.width - img.width / zoom_level);
        var y_top =  y_offset*(img.height - img.height/ zoom_level);
        ctx.drawImage(img,
            x_top ,
            y_top,
            img.width / zoom_level, img.height / zoom_level,
            0, 0, curr_canvas.width, curr_canvas.height);

        if (detections) {
            ctx.beginPath();
            var w_ratio = zoom_level*(curr_canvas.width / img.width);
            var h_ratio = zoom_level*(curr_canvas.height / img.height);
            var scale_array = [w_ratio, h_ratio, w_ratio, h_ratio];

            Object.keys(detections).map(function (bbox_id) {
                bbox = detections[bbox_id].map((value, idx) => value * scale_array[idx]);
                bbox = [bbox[0] -  x_offset*(curr_canvas.width),
                        bbox[1] - y_offset*(curr_canvas.height),
                        (bbox[2] - bbox[0]),
                        (bbox[3] - bbox[1])];
                ctx.rect(bbox[0], bbox[1], bbox[2], bbox[3]);
            });
            ctx.strokeStyle = 'yellow';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
        ctx.fillStyle = 'yellow';
        ctx.fillText(img_title, curr_canvas.width - 5 * img_title.length, curr_canvas.height -5);
    };
    img.crossOrigin = true;
    img.src = curr_canvas.getAttribute('src');
}