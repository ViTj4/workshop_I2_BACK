from flask import Flask, request, jsonify
from PIL import Image
import face_recognition
import numpy as np
import pytesseract  # Pour l'OCR
import io
from flask_cors import CORS
from transformers import pipeline
from flask_restx import Api, Resource, fields

# Charger le pipeline de classification de texte pour détecter la toxicité
classifier = pipeline("text-classification", model="unitary/toxic-bert")

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialiser Flask
app = Flask(__name__)

# CORS pour permettre les requêtes depuis le front-end
CORS(app)

# Initialiser l'API Flask-RESTX (anciennement RESTPlus)
api = Api(app, version='1.0', title='Face & Toxicity API', description='API pour la comparaison de visages et l’analyse de la toxicité')

# Définition des modèles Swagger pour l'entrée et la sortie
face_comparison = api.model('FaceComparison', {
    'same_person': fields.Boolean(description='Indique si les deux visages correspondent'),
    'face_distance': fields.Float(description='Distance de similarité entre les deux visages'),
    'name': fields.List(fields.String, description='Nom extrait de la carte d’identité')
})

message_analysis = api.model('MessageAnalysis', {
    'message': fields.String(description='Le message envoyé pour analyse'),
    'label': fields.String(description='Le label de classification, ex: toxic'),
    'score': fields.Float(description='Le score de confiance pour le label'),
    'status': fields.String(description='Statut du message basé sur le score (Toxic, Suspicious, OK)')
})

message_input = api.model('MessageInput', {
    'message': fields.String(required=True, description='Le message à analyser')
})

# Route pour comparer deux visages
@api.route('/compare_faces')
class CompareFaces(Resource):
    @api.doc('compare_faces')
    @api.expect(api.parser().add_argument('image1', location='files', type='file', required=True)
                           .add_argument('image2', location='files', type='file', required=True)
                           .add_argument('image3', location='files', type='file', required=True))
    @api.marshal_with(face_comparison)
    def post(self):
        if 'image1' not in request.files or 'image2' not in request.files or 'image3' not in request.files:
            api.abort(400, "Veuillez fournir deux images.")

        # Récupérer les deux images
        image1_data = request.files['image1'].read()
        image2_data = request.files['image2'].read()
        image3_data = request.files['image3'].read()

        # Charger les images avec face_recognition
        image1 = face_recognition.load_image_file(io.BytesIO(image1_data))
        image2 = face_recognition.load_image_file(io.BytesIO(image2_data))
        image3 = face_recognition.load_image_file(io.BytesIO(image3_data))

        # 1. Extraction du texte de la carte d'identité (image1)
        try:
            id_image_pil = Image.open(io.BytesIO(image1_data))  # Charger l'image en format PIL
            ocr_text = pytesseract.image_to_string(id_image_pil)  # Extraire le texte avec Tesseract

            # Chercher le nom dans l'OCR (à ajuster selon le format de la carte d'identité)
            lines = ocr_text.split("\n")

        except Exception as e:
            api.abort(500, f"Erreur lors de l'extraction du texte: {str(e)}")

        # Encoder les visages des deux images
        image1_face_encodings = face_recognition.face_encodings(image1)
        image2_face_encodings = face_recognition.face_encodings(image2)
        image3_face_encodings = face_recognition.face_encodings(image3)

        # Vérifier si des visages sont détectés dans les deux images
        if len(image1_face_encodings) == 0 or len(image3_face_encodings) == 0:
            api.abort(400, "Impossible de détecter un visage dans l'une des images.")

        # Prendre le premier visage trouvé dans chaque image
        image1_face_encoding = image1_face_encodings[0]
        image2_face_encoding = image2_face_encodings[0]
        image3_face_encoding = image3_face_encodings[0]

        # Comparer les visages
        results = face_recognition.compare_faces([image1_face_encoding], image3_face_encoding)
        face_distance = face_recognition.face_distance([image1_face_encoding], image3_face_encoding)[0]

        # Retourner le résultat en JSON, y compris le nom récupéré
        return {
            "same_person": bool(results[0]),
            "face_distance": float(face_distance),
            "name": lines
        }

# Route pour analyser la toxicité d'un message
@api.route('/analyze_message')
class AnalyzeMessage(Resource):
    @api.doc('analyze_message')
    @api.expect(message_input)
    @api.marshal_with(message_analysis)
    def post(self):
        data = request.get_json()  # Récupérer les données JSON envoyées par le front
        if 'message' not in data:
            api.abort(400, "Veuillez fournir un message à analyser.")

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
        return {
            "message": message,
            "label": label,
            "score": score,
            "status": status
        }

# Lancer l'application Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
