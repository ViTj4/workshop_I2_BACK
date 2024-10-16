README
Project Overview

This project is a Flask-based API for performing two main tasks:

    Face Comparison: Compares faces from two provided images and checks if they belong to the same person. It also extracts text from an ID card (OCR) to retrieve and verify the name.
    Toxicity Detection: Analyzes a given message to detect if it contains toxic content using a pre-trained BERT model.

Features

    Face Comparison:
        Compares faces in images and calculates their similarity using the face_recognition library.
        Extracts text from an ID card image using OCR with Tesseract.
        The API returns whether the faces match, the similarity score, and the extracted name from the ID card.

    Toxicity Detection:
        Uses the pre-trained unitary/toxic-bert model from Hugging Face's transformers library to analyze the toxicity of a message.
        The API returns the label (toxic or not), the confidence score, and a status (Toxic, Suspicious, or OK).

Technologies Used

    Flask: Python web framework for building APIs.
    Face Recognition: face_recognition library for facial comparison.
    OCR: pytesseract for text extraction from images.
    Text Classification: Hugging Face's transformers for toxicity detection.
    CORS: Handled by Flask-CORS to allow cross-origin resource sharing.
    Pillow (PIL): For image processing.

Setup Instructions
Requirements

    Python 3.7+
    Required libraries (can be installed via requirements.txt):
flask
flask-cors
pillow
face_recognition
transformers
torch
pytesseract

Installing Dependencies

    Clone the repository:
    git clone <repository_url>
    cd <repository_folder>

Install dependencies:
    pip install -r requirements.txt
Install Tesseract OCR. If using Windows, download and install from here. Be sure to update the path in the code:
  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
  
Running the Application

To run the Flask API:
    python app.py

By default, the application runs at http://0.0.0.0:5000/.

API Endpoints
1. /compare_faces - POST

Description: Compares two faces from provided images and extracts the name from an ID card.

Request:

    Method: POST
    Content-Type: multipart/form-data
    Required files:
        image1: ID card image (for text extraction and face comparison).
        image2: First image for face comparison.
        image3: Second image for face comparison.

Response:

    JSON response with the following fields:
        same_person: true or false based on face comparison.
        face_distance: Similarity score between the two faces.
        name: Text extracted from the ID card (can be adjusted based on format).

Example:

bash

curl -X POST -F 'image1=@id_card.jpg' -F 'image2=@face1.jpg' -F 'image3=@face2.jpg' http://localhost:5000/compare_faces

Sample Response:

json

{
  "same_person": true,
  "face_distance": 0.35,
  "name": "John Doe"
}

2. /analyze_message - POST

Description: Analyzes a message for toxic content.

Request:

    Method: POST
    Content-Type: application/json
    Required body:
        message: A text message to analyze.

Response:

    JSON response with the following fields:
        message: The original message.
        label: The classification result (toxic or not).
        score: Confidence score of the classification.
        status: Status based on the score (Toxic, Suspicious, or OK).

Example:

bash

curl -X POST -H "Content-Type: application/json" -d '{"message": "I hate you!"}' http://localhost:5000/analyze_message

Sample Response:

json

{
  "message": "I hate you!",
  "label": "toxic",
  "score": 0.98,
  "status": "Toxic"
}

Additional Notes

    The pytesseract path is system-specific. Ensure the path to the Tesseract executable is correct for your environment.
    The model used for toxicity detection (unitary/toxic-bert) can be fine-tuned or replaced as needed.
