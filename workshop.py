from flask import Flask, request, jsonify
from PIL import Image
import face_recognition
import numpy as np
import io
from flask_cors import CORS, cross_origin
from flask_talisman import Talisman


app = Flask(__name__)
Talisman(app, strict_transport_security=True, content_security_policy=None)
cors = CORS(app)
@cross_origin()
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

if __name__ == '__main__':
    app.run(debug=True)


