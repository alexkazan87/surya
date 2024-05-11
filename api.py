from flask import Flask, request, jsonify, make_response
from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection import segformer
from surya.model.recognition.model import load_model
from surya.model.recognition.processor import load_processor
from surya.input.langs import replace_lang_with_code, get_unique_langs
from surya.input.load import load_from_folder, load_from_file, load_lang_file
from surya.model.detection.segformer import load_model as load_detection_model, load_processor as load_detection_processor
from surya.model.recognition.model import load_model as load_recognition_model
from surya.model.recognition.processor import load_processor as load_recognition_processor
from surya.model.recognition.tokenizer import _tokenize
from surya.ocr import run_ocr
from surya.postprocessing.text import draw_text_on_image
from surya.settings import settings
import json
import torch

app = Flask(__name__)



# Load Surya models outside the request handler for efficiency
det_processor, det_model = segformer.load_processor(), segformer.load_model()

@app.route('/scan_image', methods=['POST'])
def scan_image():
    """
    API endpoint to scan an image and perform OCR with specified languages.

    Expects a multipart form data request with an image file in the 'image' field.
    Returns a JSON response containing the OCR results for each language.
    """
# Try to create a CUDA tensor
    try:
      device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
      print("device", device)
    except RuntimeError as e:
      if "CUDA unavailable" in str(e):
        print("CUDA is not available.")
      else:
        print("An error occurred:", e)
    print("setting", settings)

    try:
        # Validate request data
        if 'image' not in request.files:
            return make_response(jsonify({'error': 'Missing image file'}), 400)

        image_file = request.files['image']

        # Validate image format
        if not image_file.mimetype.startswith('image/'):
            return make_response(jsonify({'error': 'Invalid image format'}), 400)

        image = Image.open(image_file)
        print(f"Uploaded file path: {image_file.filename}")


        langs = request.args.getlist('langs')[0].split(",")
        print(f"Langs: {langs}")

        replace_lang_with_code(langs)

        _, lang_tokens = _tokenize("", get_unique_langs([langs]))
        rec_model = load_recognition_model(langs=lang_tokens) # Prune model moe layer to only include languages we need
        rec_processor = load_recognition_processor()



        # Perform OCR with Surya
        img_pred = run_ocr([image], [langs], det_model, det_processor, rec_model, rec_processor)[0]
        print(f"Langs: {img_pred}")



        return img_pred.model_dump()

    except Exception as e:
        # Handle unexpected errors gracefully
        print(f"An error occurred: {e}")
        return make_response(jsonify({'error': 'Internal server error'}), 500)

if __name__ == '__main__':
    app.run(debug=True)