Indian Sign Language Recognition

This project is a real-time Indian Sign Language (ISL) recognition system that uses hand gestures to recognize common words and phrases. It leverages MediaPipe for hand tracking, OpenCV for video processing, gTTS for text-to-speech conversion, and Googletrans for translation into Indian languages.

File Structure:
sign-language-recognition/
│
├── app.py
├── gesture.py
├── static/
│   ├── style.css
│   ├── script.js
│   └── train.js
├── templates/
│   ├── index.html
│   └── train.html
├── model/ (will be created automatically)
├── training_data/ (will be created automatically)
└── output/ (will be created automatically)

Features

- Real-time Gesture Recognition: Recognizes hand gestures and maps them to common words and phrases.
- Text-to-Speech: Converts recognized text into speech using gTTS.
- Translation: Translates recognized text into Indian languages (e.g., Hindi, Tamil, Telugu, etc.).
- History Log: Keeps a log of recognized gestures and saves them in a text file.
- Voice Output: Saves voice output as `.mp3` files for each recognized gesture.

Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.7 or higher
- pip (Python package manager)

Installation

1. Clone the Repository:
   git clone https://github.com/your-username/indian-sign-language-recognition.git
   cd indian-sign-language-recognition


2. Install Dependencies:
   Install the required Python packages using the `requirements.txt` file:
   
   pip install -r requirements.txt
 

3. Run the Application:
   Start the Flask application by running:
   
   python app.py
   

4. Access the Application:
   Open your web browser and navigate to:
   
   http://127.0.0.1:5000
   

Usage

1. Start Detection:
   - Click the Start button to begin gesture recognition.
   - Perform hand gestures in front of your webcam.

2. Recognized Text:
   - The recognized text will appear in the "Recognized Text" box.
   - Click the Speak button to hear the recognized text.

3. Translation:
   - Select a language from the dropdown menu.
   - Click the Translate button to translate the recognized text.
   - Click the Speak Translated Text button to hear the translated text.

4. History Log:
   - The history of recognized gestures is displayed in the "History" section.
   - Recognized text and voice output are saved in the `output` folder with timestamps.

5. Stop Detection:
   - Click the Stop button to stop gesture recognition.

Folder Structure


indian-sign-language-recognition/
├── app.py                  # Flask application
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation
├── templates/              # HTML templates
│   └── index.html          # Main web page
├── static/                 # Static files (CSS, JS, etc.)
│   └── style.css           # Stylesheet for the web page
└── output/                 # Folder for saving logs and voice output


Dependencies

- Flask: Web framework for building the application.
- OpenCV: For video capture and image processing.
- MediaPipe: For hand gesture recognition.
- gTTS: For text-to-speech conversion.
- Googletrans: For translating text into Indian languages.

Troubleshooting

1. No Audio Playback:
   - Ensure your browser allows audio playback.
   - Click the page to allow audio autoplay.

2. Gesture Recognition Issues:
   - Ensure proper lighting and clear hand gestures.
   - Adjust the camera position for better recognition.

3. Translation Errors:
   - Ensure you have an active internet connection, as Googletrans requires it.

 Acknowledgments

- MediaPipe: For providing an excellent hand tracking solution.
- gTTS: For easy text-to-speech conversion.
- Googletrans: For translation capabilities.

---

Note: This project is for educational purposes and may require further refinement for production use.

Key Sections:

1. Project Overview: Briefly describes the project and its features.
2. Prerequisites: Lists the required software and tools.
3. Installation: Provides step-by-step instructions for setting up the project.
4. Usage: Explains how to use the application.
5. Folder Structure: Describes the organization of the project files.
6. Dependencies: Lists the Python libraries used in the project.
7. Troubleshooting: Offers solutions to common issues.
8. Contributing: Explains how others can contribute to the project.
9. License: Specifies the license under which the project is distributed.
10. Acknowledgments: Credits the libraries and tools used in the project.

