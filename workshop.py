from flask import Flask, request, jsonify
from PIL import Image
import face_recognition
import numpy as np
import io
from flask_cors import CORS, cross_origin
from transformers import pipeline

# Charger le pipeline de classification de texte pour détecter la toxicité
classifier = pipeline("text-classification", model="unitary/toxic-bert")

# Initialiser Flask
app = Flask(__name__)

# CORS pour permettre les requêtes depuis le front-end
cors = CORS(app)
@cross_origin()

# Route pour comparer deux visages
@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"error": "Veuillez fournir deux images."}), 400

    # Récupérer les deux images
    image1 = request.files['image1'].read()
    image2 = request.files['image2'].read()

    # Charger les images avec face_recognition
    image1 = face_recognition.load_image_file(io.BytesIO(image1))
    image2 = face_recognition.load_image_file(io.BytesIO(image2))

    # Encoder les visages des deux images
    image1_face_encodings = face_recognition.face_encodings(image1)
    image2_face_encodings = face_recognition.face_encodings(image2)

    # Vérifier si des visages sont détectés dans les deux images
    if len(image1_face_encodings) == 0 or len(image2_face_encodings) == 0:
        return jsonify({"error": "Impossible de détecter un visage dans l'une des images."}), 400

    # Prendre le premier visage trouvé dans chaque image
    image1_face_encoding = image1_face_encodings[0]
    image2_face_encoding = image2_face_encodings[0]

    # Comparer les visages
    results = face_recognition.compare_faces([image1_face_encoding], image2_face_encoding)
    face_distance = face_recognition.face_distance([image1_face_encoding], image2_face_encoding)[0]

    # Conversion explicite pour éviter l'erreur JSON
    return jsonify({
        "same_person": bool(results[0]),  # Conversion en booléen Python natif
        "face_distance": float(face_distance)  # Conversion explicite en float (optionnel mais sûr)
    })

# Route pour analyser la toxicité d'un message
@app.route('/analyze_message', methods=['POST'])
def analyze_message():
    data = request.get_json()  # Récupérer les données JSON envoyées par le front
    if 'message' not in data:
        return jsonify({"error": "Veuillez fournir un message à analyser."}), 400
    
    message = data['message']

    # Analyser le message
    result = classifier(message)[0]
    label = result['label']
    score = result['score']

    # Interpréter le résultat basé sur le score
    if score > 0.75 and label == 'toxic':
        status = "Toxic"
    elif score > 0.50 and label == 'toxic':
        status = "Suspicious"
    else:
        status = "OK"

    # Renvoyer le résultat en JSON
    return jsonify({
        "message": message,
        "label": label,
        "score": score,
        "status": status
    })

# Lancer l'application Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
