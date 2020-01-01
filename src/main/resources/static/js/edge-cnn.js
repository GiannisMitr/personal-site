/**
 * @license
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Shamelessly used code fragments from:
 * https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd
 * https://github.com/tensorflow/tfjs-models/tree/master/body-pix
 * https://github.com/tensorflow/tfjs-models/tree/master/posenet
 * =============================================================================
 */

const state = {
    videoSupport: null,
    video: null,
    canvas: null,
    stream: null,
    modelPromise: null,
    segmentModelPromise: null,
    typeRadio: null,
    personRadio:null,
    drawSkeleton: false,
    outputStride: 16,
    mobileNetMultiplier: 0.75,
    quantBytes: 4,
    segmentationThreshold: 0.5,
    canvasOpacity: 0.7,
    scaleCanvas: null,
    videoRatio: (window.innerHeight > window.innerWidth) ? 1.33 : 0.75,
    videoContainer: null,
    isMobile: null

};
const COLOR = 'aqua';
const LINE_WIDTH = 2;


var classToColorMap = {"person": "#ff0e1d", "cup": "#76FF53", "bottle": "#6191FF", "cell phone": "#f07afa"};


function dynamicallyLoadScript(url, callback) {
    var script = document.createElement("script"); //Make a script DOM node
    script.src = url; //Set it's src to the provided URL
    script.type = 'text/javascript';
    document.body.appendChild(script);
    //Add it to the end of the head section of the page (could change 'head' to 'body' to add it to the end of the body section instead)
    if (callback != null) {
        if (script.readyState) {  // only required for IE <9
            script.onreadystatechange = function () {
                if (script.readyState === "loaded" || script.readyState === "complete") {
                    script.onreadystatechange = null;
                    callback();
                }
            };
        } else {  //Others
            script.onload = function () {
                callback();
            };
        }
    }


}

function onRadioChange() {
    document.getElementById("object-desc").style.display = state.typeRadio.checked ? 'block' : 'none';
    document.getElementById("person-desc").style.display = !state.typeRadio.checked ? 'block' : 'none';
    document.getElementById("person-options").style.display = !state.typeRadio.checked ? 'block' : 'none';

    updateModel()
}

function init() {

    if (state.videoSupport) {
        document.getElementById("discl").style.visibility = "hidden";
        state.canvas.style.display = "block";
        state.video.style.display = "block";
        const webCamPromise = loadWebCam();

        webCamPromise.then(updateModel)
    }
}

function loadWebCam() {
    return navigator.mediaDevices
        .getUserMedia({
            audio: false,
            video: {
                facingMode: "user"
            }
        })
        .then(stream => {
            window.stream = stream;
            state.video.srcObject = stream;
            return new Promise((resolve, reject) => {
                state.video.onloadedmetadata = () => {
                    state.videoRatio = state.video.videoHeight / state.video.videoWidth;
                    resolve();
                };
            });
        });
}


function updateModel() {
    if (state.video.srcObject == null) {
        //if video didn't start then return
        return;
    }
    if (state.animId != null) {
        cancelAnimationFrame(state.animId);
        state.canvas.getContext("2d").clearRect(0, 0, state.canvas.width, state.canvas.height);
    }

    if (state.typeRadio.checked) {
        //object detection requested, load modelPromise if not already loaded
        state.modelPromise.then(model => {
            //in case the bodyPix modelPromise set it hidden before
            state.video.style.visibility = 'visible'
            this.detectFrame(state.video, model);
        }).catch(error => {
            console.error(error);
        });
    } else {
        if (state.segmentModelPromise == null) {
            //give it some time to load
            setTimeout(updateModel, 500);
            return;
        }
        state.segmentModelPromise.then(model => {
            //no need to re-render video the mask drawn includes it
            state.video.style.visibility = 'hidden'
            this.estimateAndDrawPose(state.video, model);
        });
    }
}


function estimateAndDrawPose(frame, model) {
    state.scaleCanvas.getContext('2d').drawImage(frame, 0, 0, state.video.width, state.video.height);

    (state.isMobile ? model.segmentPersonParts(state.scaleCanvas) : model.segmentMultiPersonParts(state.scaleCanvas))
        .then(predictions => {
            var coloredPartImage = bodyPix.toColoredPartMask(predictions);

            // draw the colored part image on top of the original image onto a canvas.  The colored part image will be drawn semi-transparent, with an opacity of 0.7, allowing for the original image to be visible under.

            bodyPix.drawMask(state.canvas, state.scaleCanvas, coloredPartImage, state.canvasOpacity, 0, false);
            if (state.drawSkeleton) {
                drawPoses(predictions, false, state.canvas.getContext('2d'));
            }
            state.animId = requestAnimationFrame(() => {
                this.estimateAndDrawPose(frame, model);
            });
        });

}


detectFrame = (frame, model) => {
    model.detect(frame).then(predictions => {

        this.renderPredictions(predictions);
        state.animId = requestAnimationFrame(() => {
            this.detectFrame(frame, model);
        });
    });
};

renderPredictions = predictions => {
    const ctx = state.canvas.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    // Font options.
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";

    predictions.forEach(prediction => {
        //make sure each class is assigned the same color every time
        var color = classToColorMap[prediction.class];
        if (color == null) {
            color = "#" + ((1 << 24) * Math.random() | 0).toString(16);
            classToColorMap[prediction.class] = color;
        }
        xTransform = state.video.width / state.video.videoWidth;
        yTransform = state.video.height / state.video.videoHeight;
        const x = prediction.bbox[0] * xTransform;
        const y = prediction.bbox[1] * yTransform;
        const width = prediction.bbox[2] * xTransform;
        const height = prediction.bbox[3] * yTransform;
        // Draw the bounding box.
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);
        // Draw the label background.
        ctx.fillStyle = color
        const textWidth = ctx.measureText(prediction.class + ': 99%').width;
        const textHeight = parseInt(font, 10); // base 10
        ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
    });

    predictions.forEach(prediction => {
        const x = prediction.bbox[0] * xTransform;
        const y = prediction.bbox[1] * yTransform;
        // Draw the text last to ensure it's on top.
        ctx.fillStyle = "#000000";
        ctx.fillText(prediction.class + ': ' + (prediction.score * 100).toFixed(0) + '%', x, y);
    });
};

function drawPoses(personOrPersonPartSegmentation, flipHorizontally, ctx) {
    if (Array.isArray(personOrPersonPartSegmentation)) {
        personOrPersonPartSegmentation.forEach(personSegmentation => {
            let pose = personSegmentation.pose;
            if (flipHorizontally) {
                pose = bodyPix.flipPoseHorizontal(pose, personSegmentation.width);
            }
            drawKeypoints(pose.keypoints, 0.1, ctx);
            drawSkeleton(pose.keypoints, 0.1, ctx);
        });
    } else {
        personOrPersonPartSegmentation.allPoses.forEach(pose => {
            if (flipHorizontally) {
                pose = bodyPix.flipPoseHorizontal(
                    pose, personOrPersonPartSegmentation.width);
            }
            drawKeypoints(pose.keypoints, 0.1, ctx);
            drawSkeleton(pose.keypoints, 0.1, ctx);
        })
    }
}

function drawPoint(ctx, y, x, r, color) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
}

/**
 * Draws a line on a canvas, i.e. a joint
 */
function drawSegment([ay, ax], [by, bx], color, scale, ctx) {
    ctx.beginPath();
    ctx.moveTo(ax * scale, ay * scale);
    ctx.lineTo(bx * scale, by * scale);
    ctx.lineWidth = LINE_WIDTH;
    ctx.strokeStyle = color;
    ctx.stroke();
}

/**
 * Draws a pose skeleton by looking up all adjacent keypoints/joints
 */
function drawSkeleton(keypoints, minConfidence, ctx, scale = 1) {
    const adjacentKeyPoints =
        posenet.getAdjacentKeyPoints(keypoints, minConfidence);

    function toTuple({y, x}) {
        return [y, x];
    }

    adjacentKeyPoints.forEach((keypoints) => {
        drawSegment(
            toTuple(keypoints[0].position), toTuple(keypoints[1].position), COLOR,
            scale, ctx);
    });
}

/**
 * Draw pose keypoints onto a canvas
 */
function drawKeypoints(keypoints, minConfidence, ctx, scale = 1) {
    for (let i = 0; i < keypoints.length; i++) {
        const keypoint = keypoints[i];

        if (keypoint.score < minConfidence) {
            continue;
        }

        const {y, x} = keypoint.position;
        drawPoint(ctx, y * scale, x * scale, 3, COLOR);
    }
}

function isAndroid() {
    return /Android/i.test(navigator.userAgent);
}

function isiOS() {
    return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
    return isAndroid() || isiOS() || isWindows();
}

function isWindows() {
    return /windows phone/i.test(navigator.userAgent);
}

function scaleVideoOnResize() {

    state.video.width = state.video.srcObject != null ? state.video.videoWidth : 640;
    state.video.height = state.video.srcObject != null ? state.video.videoHeight : 480;
    state.canvas.width = state.video.width;
    state.canvas.height = state.video.height;
    state.scaleCanvas.width = state.video.width;
    state.scaleCanvas.height = state.video.height;
    state.videoContainer.style.width = state.video.width + 'px';
    state.videoContainer.style.height = (state.video.height + 50) + 'px';
    if (document.documentElement.clientWidth < 680) {
        state.canvas.width = document.documentElement.clientWidth - 32;
        state.scaleCanvas.width = state.canvas.width;
        state.canvas.height = state.videoRatio * state.canvas.width
        state.scaleCanvas.height = state.canvas.height;
        state.video.width = document.documentElement.clientWidth - 32;
        state.video.height = state.videoRatio * state.video.width;
        state.videoContainer.style.width = (document.documentElement.clientWidth - 32) + 'px';
        state.videoContainer.style.height = (state.video.height + 50) + 'px';
    }

}

function loadBodypixModel() {
    state.segmentModelPromise = bodyPix.load({
        architecture: 'MobileNetV1',
        outputStride: 16,
        multiplier: state.mobileNetMultiplier,//can be 0.75
        quantBytes: state.quantBytes
    });//and 2
}

(function () {
    state.videoSupport = navigator.mediaDevices && navigator.mediaDevices.getUserMedia;

    document.addEventListener('DOMContentLoaded', function () {
        //load scripts on startup
        if (state.videoSupport) {
            dynamicallyLoadScript("https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd", function () {
                state.modelPromise = cocoSsd.load();
                state.modelPromise.then(function () {
                    <!-- Load BodyPix -->
                    dynamicallyLoadScript("https://cdn.jsdelivr.net/npm/@tensorflow-models/body-pix@2.0", loadBodypixModel);
                    dynamicallyLoadScript("https://cdn.jsdelivr.net/npm/@tensorflow-models/posenet");
                })
            });
        }
        else {
            document.getElementById("webcam-support").style.display = 'block';
            document.getElementById("discl").style.visibility = "hidden";
        }

        document.getElementById("skeleton").addEventListener('change', function () {
            state.drawSkeleton = this.checked;
        });
        state.isMobile = isMobile();
        state.video = document.getElementById("vid");
        state.videoContainer = document.getElementById("vid-cont");
        state.canvas = document.getElementById("canvas");
        state.typeRadio = document.getElementById("object");
        state.personRadio = document.getElementById("person");
        state.mobileNetMultiplier = state.isMobile ? 0.5 : 0.75;
        state.quantBytes = state.isMobile ? 2 : 4;
        state.scaleCanvas = document.createElement('canvas');
        state.scaleCanvas.width = state.canvas.width;
        state.scaleCanvas.height = state.canvas.height;
        console.log(window.location.hash.substring(1));
        if(window.location.hash && window.location.hash.substring(1)==="segment") {
            state.personRadio.click();
        }
        scaleVideoOnResize();

        window.addEventListener('resize', function () {
            scaleVideoOnResize();
            if (!state.isMobile) {
                updateModel();
            }
        });
        window.addEventListener("orientationchange", function () {
            scaleVideoOnResize();
            updateModel();
        });
    }, false);

})();


