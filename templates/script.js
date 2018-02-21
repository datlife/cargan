STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen',
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

function drawImage() {
    var overviews = document.querySelectorAll("#overview canvas");
    var buttons = document.querySelectorAll('button:not(.close)');
    overviews.forEach(
        function (curr_canvas, curr_idx, listObj) {
            _draw_initial_frame(curr_canvas);
            curr_canvas.addEventListener('mouseenter', function () {
                _loop_images(curr_canvas)
            }, false);
        }
    );
    buttons.forEach(
        function (button, idx, lst) {
            button.addEventListener('click', function () {
                var seq_id = button.getAttribute('data-target');
                var canvas_list = document.querySelectorAll(seq_id + " canvas");
                canvas_list.forEach(
                    function (curr_canvas, curr_idx, listObj) {
                        _draw_image(
                            curr_canvas,
                            curr_canvas.getAttribute('src'),
                            JSON.parse(curr_canvas.getAttribute('detections').replace('/', '')).bboxes,
                            JSON.parse(curr_canvas.getAttribute('detections')).scores,
                            0, 0, 1.0);
                    }
                );
            })
        }
    )

}

function _loop_images(canvas) {
    var FPS = 1000/20;
    var images = JSON.parse(canvas.getAttribute('frames'));
    var detections = JSON.parse(canvas.getAttribute('detections').replace('/', ''));
    _draw_imgs_list(canvas, images, detections, FPS); // actually draw things
}

function _draw_imgs_list(canvas, images, detections, FPS) {
    var breaksignal = false;
    var keys = Object.keys(images);
    for (let img_id of keys){
        function doLoop(){
            if (breaksignal === false) {
                var img_url = images[img_id];
                var detection = detections[img_url.split('/').pop()];
                if (detection) {
                    _draw_image(canvas, img_url, detection.bboxes, detection.scores, 0, 0, 1.0);
                }
                else {
                    _draw_image(canvas, img_url, null, null, 0, 0, 1.0);
                }
            }
        }
        setTimeout(doLoop, FPS * img_id);
    }
    $(canvas).on("mouseout", function (e) {
        breaksignal = true;
    });

}

function _draw_initial_frame(canvas){
    var images = JSON.parse(canvas.getAttribute('frames'));
    var detections = JSON.parse(canvas.getAttribute('detections').replace('/', ''));
    //draw first frame
    var img_url = images[0];
    var detection = detections[img_url.split('/').pop()];
    if (detection) {
        _draw_image(canvas, img_url, detection.bboxes, detection.scores, 0, 0, 1.0);
    }
    else {
        _draw_image(canvas, img_url, null, null, 0, 0, 1.0);
    }
}


function _draw_image(canvas, img_url, bboxes, confidence_scores) {
    var ctx = canvas.getContext("2d");
    var img_title =img_url.split('/').pop();
    var img = new Image();

    img.onload = function () {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        var x_top =  img.width - img.width;
        var y_top =  img.height - img.height;

        ctx.drawImage(img,
            x_top ,
            y_top,
            img.width, img.height,
            0, 0, canvas.width, canvas.height);

        if (bboxes) {
            var w_ratio = (canvas.width / img.width);
            var h_ratio = (canvas.height / img.height);
            var scale_array = [w_ratio, h_ratio, w_ratio, h_ratio];
            Object.keys(bboxes).map(function (bbox_id) {
                ctx.beginPath();
                bbox = bboxes[bbox_id].map((value, idx) => value * scale_array[idx]);
                xpos = bbox[0];
                ypos = bbox[1];
                width = bbox[2] - bbox[0];
                height = bbox[3] - bbox[1];

                ctx.rect(xpos, ypos, width, height);
                object_id = Math.round(confidence_scores[bbox_id])
                drawTextBG(ctx, object_id, '8px arial',  xpos, ypos-9, STANDARD_COLORS[object_id]);
                ctx.strokeStyle =  STANDARD_COLORS[object_id];
                ctx.lineWidth = 3;
                ctx.stroke();
            });

        }
        ctx.fillStyle = 'yellow';
        ctx.fillText(img_title, canvas.width - 5 * img_title.length, canvas.height -5);
    };
    img.crossOrigin = true;
    img.src = img_url;
}

function drawTextBG(ctx, txt, font, x, y, color) {
    ctx.save();
    ctx.font = font;
    ctx.textBaseline = 'top';
    ctx.fillStyle = color;
    var width = ctx.measureText(txt).width;
    ctx.fillRect(x, y, width, parseInt(font, 10));
    ctx.fillStyle = 'black';
    ctx.fillText(txt, x, y);
    ctx.restore();
}

function random_color(number){

}