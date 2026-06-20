const socket = io();

const videoElement = document.getElementById("video");
const canvasElement = document.getElementById("canvas");
const canvasCtx = canvasElement.getContext("2d");

const predictionDisplay = document.getElementById("prediction");
const sentenceDisplay = document.getElementById("sentence");
const cameraBtn = document.getElementById("cameraBtn");

let cameraActive = false;
let camera = null;
let lastSentTime = 0;

const hands = new Hands({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
}});

hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

hands.onResults(onResults);

function onResults(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const landmarks = results.multiHandLandmarks[0];

    drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 4});
    drawLandmarks(canvasCtx, landmarks, {color: '#FF0000', lineWidth: 2});

    const xValues = landmarks.map(l => l.x);
    const yValues = landmarks.map(l => l.y);
    const x1 = Math.min(...xValues) * canvasElement.width - 20;
    const y1 = Math.min(...yValues) * canvasElement.height - 20;
    const width = (Math.max(...xValues) * canvasElement.width + 20) - x1;
    const height = (Math.max(...yValues) * canvasElement.height + 20) - y1;

    canvasCtx.strokeStyle = "#00FF00";
    canvasCtx.lineWidth = 3;
    canvasCtx.strokeRect(x1, y1, width, height);

    const now = Date.now();
    if (now - lastSentTime > 100) {
        socket.emit("process_landmarks", { landmarks: landmarks });
        lastSentTime = now;
    }
  }
  canvasCtx.restore();
}

function toggleCamera() {
    if (!cameraActive) {
        camera = new Camera(videoElement, {
            onFrame: async () => {
                await hands.send({image: videoElement});
            },
            width: 640,
            height: 480
        });

        camera.start()
        .then(() => {
            cameraActive = true;
            cameraBtn.innerText = "Stop Camera";
            cameraBtn.classList.add("active");
            canvasElement.width = 640;
            canvasElement.height = 480;
        })
        .catch(err => {
            console.error("Camera Error:", err);
            alert("Please allow camera access.");
        });

    } else {
        if (camera) {
            window.location.reload(); 
        }
    }
}

socket.on("prediction", (data) => predictionDisplay.innerText = data.prediction || "-");
socket.on("sentence_update", (data) => sentenceDisplay.innerText = data.sentence || "");
socket.on("stable_prediction", (data) => {}); 

document.getElementById('clear-btn').onclick = () => socket.emit('clear_sentence');
document.getElementById('backspace-btn').onclick = () => socket.emit('backspace');