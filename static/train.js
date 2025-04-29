document.addEventListener('DOMContentLoaded', () => {
    // Webcam training elements
    const startWebcamTrainingBtn = document.getElementById('start-webcam-training');
    const stopWebcamTrainingBtn = document.getElementById('stop-webcam-training');
    const webcamLabelInput = document.getElementById('webcam-label');
    const countdownElement = document.getElementById('countdown');
    const countdownNumber = document.getElementById('countdown-number');
    const webcamStatus = document.getElementById('webcam-training-status');
    
    // Model training elements
    const trainModelBtn = document.getElementById('train-model');
    const labelsCountElement = document.getElementById('labels-count');
    const modelTrainingInfo = document.getElementById('model-training-info');
    
    // Other elements
    const uploadTrainingBtn = document.getElementById('upload-training');
    const uploadLabelInput = document.getElementById('upload-label');
    const trainingFileInput = document.getElementById('training-file');
    const labelsList = document.getElementById('labels-list');
    const refreshLabelsBtn = document.getElementById('refresh-labels');
    
    let countdownInterval;
    let isTraining = false;
    
    // Initialize the page
    loadTrainingLabels();
    
    // Webcam training with countdown
    startWebcamTrainingBtn.addEventListener('click', () => {
        const label = webcamLabelInput.value.trim();
        if (!label) {
            alert('Please enter a label');
            return;
        }
        
        // Show countdown
        countdownElement.style.display = 'block';
        startWebcamTrainingBtn.style.display = 'none';
        stopWebcamTrainingBtn.style.display = 'inline-block';
        
        let count = 3;
        countdownNumber.textContent = count;
        webcamStatus.textContent = 'Preparing to train...';
        
        countdownInterval = setInterval(() => {
            count--;
            countdownNumber.textContent = count;
            
            if (count <= 0) {
                clearInterval(countdownInterval);
                countdownElement.style.display = 'none';
                startTraining(label);
            }
        }, 1000);
    });
    
    // Stop webcam training
    stopWebcamTrainingBtn.addEventListener('click', () => {
        clearInterval(countdownInterval);
        countdownElement.style.display = 'none';
        startWebcamTrainingBtn.style.display = 'inline-block';
        stopWebcamTrainingBtn.style.display = 'none';
        webcamStatus.textContent = 'Training stopped';
        
        fetch('/stop_training', {
            method: 'POST'
        }).then(() => {
            isTraining = false;
        });
    });
    
    function startTraining(label) {
        isTraining = true;
        webcamStatus.textContent = `Training for "${label}" in progress...`;
        
        fetch('/start_webcam_training', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ label: label })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                webcamStatus.textContent = `Training for "${label}" is active`;
            } else {
                webcamStatus.textContent = 'Training failed: ' + data.message;
                startWebcamTrainingBtn.style.display = 'inline-block';
                stopWebcamTrainingBtn.style.display = 'none';
            }
        });
    }
    
    // Upload training files
    uploadTrainingBtn.addEventListener('click', () => {
        const label = uploadLabelInput.value.trim();
        if (!label) {
            alert('Please enter a label');
            return;
        }
        
        const files = trainingFileInput.files;
        if (files.length === 0) {
            alert('Please select files to upload');
            return;
        }
        
        const formData = new FormData();
        formData.append('label', label);
        for (let i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }
        
        uploadTrainingBtn.disabled = true;
        uploadTrainingBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';
        
        fetch('/upload_training_files', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                alert(`Uploaded ${data.count} files for "${label}"`);
                loadTrainingLabels();
                trainingFileInput.value = '';
            } else {
                alert(data.message);
            }
        })
        .finally(() => {
            uploadTrainingBtn.disabled = false;
            uploadTrainingBtn.innerHTML = '<i class="fas fa-upload"></i> Upload Files';
        });
    });
    
    // Train model button with conditional styling
    trainModelBtn.addEventListener('click', trainModel);
    
    // Refresh labels button
    refreshLabelsBtn.addEventListener('click', loadTrainingLabels);
    
    // Load existing labels and update UI
    function loadTrainingLabels() {
        fetch('/get_training_labels')
            .then(response => response.json())
            .then(data => {
                labelsList.innerHTML = '';
                data.labels.forEach(label => {
                    const li = document.createElement('li');
                    li.textContent = label;
                    labelsList.appendChild(li);
                });
                
                // Update label count and train button state
                const labelCount = data.labels.length;
                labelsCountElement.textContent = labelCount;
                
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
    
    // Train model function
    function trainModel() {
        const labelCount = parseInt(labelsCountElement.textContent);
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
            // First check if the response is JSON
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
                <p><strong>Total Models:</strong> ${data.total_models}</p>
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
});
