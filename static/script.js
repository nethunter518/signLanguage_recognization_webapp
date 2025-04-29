// DOM Elements
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const clearBtn = document.getElementById('clear-btn');
const gestureText = document.getElementById('gesture-text');
const historyList = document.getElementById('history-list');
const translateBtn = document.getElementById('translate-btn');
const translatedText = document.getElementById('translated-text');
const languageSelect = document.getElementById('language-select');
const statusIndicator = document.getElementById('status-indicator');
const videoStats = document.getElementById('video-stats');
const downloadTextBtn = document.getElementById('download-text-btn');
const downloadAudioBtn = document.getElementById('download-audio-btn');
const downloadTranslatedTextBtn = document.getElementById('download-translated-text-btn');
const downloadTranslatedAudioBtn = document.getElementById('download-translated-audio-btn');
const webcamModeBtn = document.getElementById('webcam-mode-btn');
const videoUploadModeBtn = document.getElementById('video-upload-mode-btn');
const imageUploadModeBtn = document.getElementById('image-upload-mode-btn');
const fileUploadContainer = document.getElementById('file-upload-container');
const fileInput = document.getElementById('file-input');
const browseFilesBtn = document.getElementById('browse-files-btn');
const selectedFileName = document.getElementById('selected-file-name');
const startTrainingBtn = document.getElementById('start-training-btn');
const uploadTrainingBtn = document.getElementById('upload-training-btn');
const gestureLabelSelect = document.getElementById('gesture-label');
const trainingProgress = document.getElementById('training-progress');
const progressText = document.getElementById('progress-text');
const trainModelBtn = document.getElementById('train-model-btn');

// State variables
let gestureHistory = [];
let lastRecognizedText = "";
let detectionInterval;
let handCount = 0;
let confidenceLevel = 0;
let currentInputMode = 'webcam';
let videoElement = null;
let imageElement = null;
let trainingMode = false;
let frameProcessing = false;

// Initialize button states
stopBtn.disabled = true;

// Event Listeners
startBtn.addEventListener('click', startDetection);
stopBtn.addEventListener('click', stopDetection);
clearBtn.addEventListener('click', clearText);
translateBtn.addEventListener('click', translateText);
downloadTextBtn.addEventListener('click', () => downloadText(gestureText.value, 'output'));
downloadAudioBtn.addEventListener('click', () => downloadAudio(gestureText.value, 'output'));
downloadTranslatedTextBtn.addEventListener('click', () => downloadText(translatedText.value, 'translation'));
downloadTranslatedAudioBtn.addEventListener('click', () => downloadAudio(translatedText.value, 'translation'));
webcamModeBtn.addEventListener('click', () => switchInputMode('webcam'));
videoUploadModeBtn.addEventListener('click', () => switchInputMode('video'));
imageUploadModeBtn.addEventListener('click', () => switchInputMode('image'));
browseFilesBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileUpload);
startTrainingBtn.addEventListener('click', toggleTrainingMode);
uploadTrainingBtn.addEventListener('click', () => {
    const label = gestureLabelSelect.value;
    if (!label) {
        alert("Please select a gesture label first");
        return;
    }
    fileInput.click();
});
trainModelBtn.addEventListener('click', trainModel);

// Functions
function startDetection() {
    fetch('/start_detection', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
    }).then(() => {
        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusIndicator.classList.remove('status-inactive');
        statusIndicator.classList.add('status-active');
        startGestureDetection();
    });
}

function stopDetection() {
    fetch('/stop_detection', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
    }).then(() => {
        startBtn.disabled = false;
        stopBtn.disabled = true;
        statusIndicator.classList.remove('status-active');
        statusIndicator.classList.add('status-inactive');
        stopGestureDetection();
    });
}

function clearText() {
    gestureText.value = "";
    translatedText.value = "";
}

function translateText() {
    const text = gestureText.value.trim();
    const lang = languageSelect.value;
    if (text) {
        fetch('/translate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text, lang: lang }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === "success") {
                translatedText.value = data.translated_text;
            }
        });
    } else {
        alert("No recognized text to translate.");
    }
}

function textToSpeech(text) {
    if (!text) return;
    
    fetch('/text_to_speech', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === "success") {
            const audio = new Audio(`/output/${data.file}`);
            audio.play().catch(() => {
                alert("Click the page to allow audio playback.");
            });
        }
    });
}

function downloadText(text, type) {
    if (!text.trim()) {
        alert(`No ${type} text to download.`);
        return;
    }
    
    fetch('/download_text', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === "success") {
            const link = document.createElement('a');
            link.href = `/output/${data.filename}`;
            link.download = data.filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    });
}

function downloadAudio(text, type) {
    if (!text.trim()) {
        alert(`No ${type} text to convert to audio.`);
        return;
    }
    
    fetch('/download_audio', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === "success") {
            const link = document.createElement('a');
            link.href = `/output/${data.filename}`;
            link.download = data.filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    });
}

function startGestureDetection() {
    fetchGestureText();
    detectionInterval = setInterval(fetchGestureText, 500);
}

function stopGestureDetection() {
    clearInterval(detectionInterval);
    updateVideoStats(false, 0, 0);
}

function fetchGestureText() {
    if (frameProcessing) return;
    frameProcessing = true;
    
    fetch('/get_gesture_text')
        .then(response => response.json())
        .then(data => {
            updateGestureText(data.text, data.confidence);
            frameProcessing = false;
        })
        .catch(() => {
            frameProcessing = false;
        });
}

function updateGestureText(text, confidence) {
    confidenceLevel = confidence || 0;
    handCount = text.includes("(") ? 2 : (text !== "Unknown Gesture" ? 1 : 0);
    updateVideoStats(true, handCount, confidenceLevel);
    
    if (text !== lastRecognizedText && text !== "Unknown Gesture") {
        lastRecognizedText = text;
        
        if (gestureText.value.trim() === "") {
            gestureText.value = text;
        } else {
            gestureText.value += " " + text;
        }
        
        const now = new Date();
        const timeString = now.toLocaleTimeString();
        gestureHistory.push({text, confidence, time: timeString});
        
        const listItem = document.createElement('li');
        listItem.innerHTML = `
            <span>${text} (${confidence}%)</span>
            <span class="history-time">${timeString}</span>
        `;
        historyList.appendChild(listItem);
        
        historyList.scrollTop = historyList.scrollHeight;
        textToSpeech(text);
    }
}

function updateVideoStats(active, hands, confidence) {
    const status = active ? "Active" : "Inactive";
    videoStats.textContent = `Detection: ${status} | Hands: ${hands} | Confidence: ${confidence}%`;
}

function switchInputMode(mode) {
    currentInputMode = mode;
    
    webcamModeBtn.classList.remove('active');
    videoUploadModeBtn.classList.remove('active');
    imageUploadModeBtn.classList.remove('active');
    
    if (mode === 'webcam') {
        webcamModeBtn.classList.add('active');
        fileUploadContainer.style.display = 'none';
        stopFileProcessing();
        startWebcam();
    } else if (mode === 'video') {
        videoUploadModeBtn.classList.add('active');
        fileUploadContainer.style.display = 'flex';
        stopWebcam();
    } else if (mode === 'image') {
        imageUploadModeBtn.classList.add('active');
        fileUploadContainer.style.display = 'flex';
        stopWebcam();
    }
}

function startWebcam() {
    document.querySelector('.video-feed img').style.display = 'block';
}

function stopWebcam() {
    document.querySelector('.video-feed img').style.display = 'none';
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    selectedFileName.textContent = file.name;
    
    if (currentInputMode === 'video') {
        processVideoFile(file);
    } else if (currentInputMode === 'image') {
        processImageFile(file);
    } else if (trainingMode) {
        uploadTrainingData(file);
    }
}

function processVideoFile(file) {
    if (!videoElement) {
        videoElement = document.createElement('video');
        videoElement.style.width = '100%';
        videoElement.style.maxWidth = '480px';
        videoElement.style.borderRadius = '8px';
        videoElement.style.boxShadow = '0 0 15px rgba(0, 0, 0, 0.3)';
        document.querySelector('.video-feed').appendChild(videoElement);
    }
    
    const videoURL = URL.createObjectURL(file);
    videoElement.src = videoURL;
    videoElement.controls = true;
    videoElement.loop = true;
    
    videoElement.addEventListener('loadedmetadata', () => {
        videoElement.currentTime = 0;
    });
    
    videoElement.addEventListener('seeked', processVideoFrame);
    videoElement.addEventListener('play', () => {
        processVideoFrame();
    });
    
    videoElement.play().catch(() => {});
}

function processVideoFrame() {
    if (!videoElement.paused && !videoElement.ended) {
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        
        processFrame(canvas.toDataURL('image/jpeg'));
        
        if (!videoElement.paused && !videoElement.ended) {
            requestAnimationFrame(processVideoFrame);
        }
    }
}

function processImageFile(file) {
    if (!imageElement) {
        imageElement = document.createElement('img');
        imageElement.style.width = '100%';
        imageElement.style.maxWidth = '480px';
        imageElement.style.borderRadius = '8px';
        imageElement.style.boxShadow = '0 0 15px rgba(0, 0, 0, 0.3)';
        document.querySelector('.video-feed').appendChild(imageElement);
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
        imageElement.src = e.target.result;
        processFrame(e.target.result);
    };
    reader.readAsDataURL(file);
}

function processFrame(imageData) {
    fetch('/process_frame', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData })
    })
    .then(response => response.json())
    .then(data => {
        updateGestureText(data.text, data.confidence);
    });
}

function stopFileProcessing() {
    if (videoElement) {
        videoElement.pause();
        videoElement.src = '';
        videoElement.remove();
        videoElement = null;
    }
    
    if (imageElement) {
        imageElement.src = '';
        imageElement.remove();
        imageElement = null;
    }
    
    selectedFileName.textContent = 'No file selected';
    fileInput.value = '';
}

function toggleTrainingMode() {
    const label = gestureLabelSelect.value;
    if (!label) {
        alert("Please select a gesture label first");
        return;
    }
    
    if (!trainingMode) {
        fetch('/start_webcam_training', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ label: label })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === "success") {
                trainingMode = true;
                startTrainingBtn.classList.add('danger');
                startTrainingBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Training';
            }
        });
    } else {
        fetch('/stop_training', {
            method: 'POST'
        })
        .then(() => {
            trainingMode = false;
            startTrainingBtn.classList.remove('danger');
            startTrainingBtn.innerHTML = '<i class="fas fa-video"></i> Webcam Training';
        });
    }
}

function uploadTrainingData(file) {
    const label = gestureLabelSelect.value;
    const formData = new FormData();
    formData.append('file', file);
    formData.append('label', label);
    
    fetch('/upload_training_files', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === "success") {
            alert(`Successfully uploaded training data for ${label}`);
            selectedFileName.textContent = 'No file selected';
            fileInput.value = '';
        }
    });
}

function populateGestureLabels() {
    const labels = [
        "Hello", "Yes", "No", "Thank You", "Please", "Goodbye", 
        "Help", "Food", "Water", "Sorry", "I Love You", "Friend",
        "Home", "School", "Work", "Name", "Age", "Where", "How", "Why",
        "Time", "Day", "Night", "Week", "Month", "Year", "Now", "Today",
        "Tomorrow", "Yesterday", "Morning", "Evening", "Afternoon", "Happy",
        "Sad", "Family", "Brother", "Sister", "Father", "Mother"
    ];
    
    gestureLabelSelect.innerHTML = '<option value="">Select Gesture Label</option>';
    labels.forEach(label => {
        const option = document.createElement('option');
        option.value = label;
        option.textContent = label;
        gestureLabelSelect.appendChild(option);
    });
}

function trainModel() {
    const labelCount = document.querySelectorAll('#labels-list li').length;
    if (labelCount < 5) {
        alert('You need at least 5 different labels to train a model');
        return;
    }

    const progress = document.getElementById('training-progress');
    const progressText = document.getElementById('progress-text');
    const trainingResults = document.getElementById('training-results');

    progress.style.display = 'flex';
    progress.value = 0;
    progressText.textContent = '0%';
    trainingResults.innerHTML = '';
    
    trainModelBtn.disabled = true;
    trainModelBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training...';

    fetch('/train_model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    })
    .then(response => {
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            return response.text().then(text => {
                throw new Error(`Expected JSON but got: ${text}`);
            });
        }
        return response.json();
    })
    .then(data => {
        if (data.status === 'success') {
            progress.value = 100;
            progressText.textContent = '100%';
            
            trainingResults.innerHTML = `
                <h3>Training Results</h3>
                <p><strong>Model Name:</strong> model${data.model_number}.h5</p>
                <p><strong>Accuracy:</strong> ${(data.accuracy * 100).toFixed(2)}%</p>
                <p><strong>Validation Accuracy:</strong> ${(data.val_accuracy * 100).toFixed(2)}%</p>
                <p><strong>Labels:</strong> ${data.labels.join(', ')}</p>
            `;
            
            setTimeout(() => {
                alert(`Model trained successfully as model${data.model_number}.h5!\nAccuracy: ${(data.accuracy * 100).toFixed(2)}%`);
            }, 1000);
        } else {
            const errorMsg = data.message || 'Training failed without error message';
            trainingResults.innerHTML = `<p class="error">${errorMsg}</p>`;
            alert(`Training failed: ${errorMsg}`);
        }
    })
    .catch(error => {
        trainingResults.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        console.error('Training error:', error);
        alert(`Error during training: ${error.message}`);
    })
    .finally(() => {
        trainModelBtn.disabled = false;
        trainModelBtn.innerHTML = '<i class="fas fa-cogs"></i> Train Model';
        loadTrainingLabels();
    });
}

function loadTrainingLabels() {
    fetch('/get_training_labels')
        .then(response => response.json())
        .then(data => {
            const labelsList = document.getElementById('labels-list');
            labelsList.innerHTML = '';
            data.labels.forEach(label => {
                const li = document.createElement('li');
                li.textContent = label;
                labelsList.appendChild(li);
            });
            
            const labelCount = data.labels.length;
            document.getElementById('labels-count').textContent = labelCount;
            
            const trainModelBtn = document.getElementById('train-model-btn');
            const modelTrainingInfo = document.getElementById('model-training-info');
            
            if (labelCount >= 5) {
                trainModelBtn.classList.remove('secondary');
                trainModelBtn.classList.add('success');
                modelTrainingInfo.textContent = 'Ready to train a new model!';
            } else {
                trainModelBtn.classList.remove('success');
                trainModelBtn.classList.add('secondary');
                modelTrainingInfo.textContent = `Minimum 5 labels needed (currently ${labelCount})`;
            }
        });
}

// Initialize
populateGestureLabels();
loadTrainingLabels();
switchInputMode('webcam');

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    fetch('/stop_detection', {
        method: 'POST'
    });
});
