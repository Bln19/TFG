from flask import Flask, render_template, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import logging
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound

app = Flask(__name__)

# Configuramos logger para ver debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variable global para almacenar los subtítulos
transcript_data = []

# obtenemos el modelo y  el tokenizador
model_save_path = './correct_transcription_model_base_v5'
tokenizer_save_path = './correct_transcription_tokenizer_base_v5'

# Cargamos el modelo y el tokenizador
tokenizer = T5Tokenizer.from_pretrained(tokenizer_save_path)
model = T5ForConditionalGeneration.from_pretrained(model_save_path)

# Definimos el tipo de dispositivo (para este modelo quizá seria mejor gpu? revisar)
device = torch.device('cpu')
model.to(device)

# Ruta principal donde se cargara el index
@app.route('/')
def index():
    return render_template('index.html')

# Obtenemos subtítulos a traves de la siguiente ruta:
@app.route('/get_subtitles', methods=['POST'])
def get_subtitles():
    global transcript_data
    # se obtiene el id del video y el idioma desde el frontend
    video_id = request.json['video_id']
    language = request.json.get('language', 'es')

    try:
        # desde la api, se obtienene los subtitulos del video
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
    except NoTranscriptFound as e:
        return jsonify({'status': 'error', 'message': str(e)}), 404

    # Guardamos los subtitulos en la variable global
    transcript_data = transcript
    return jsonify({'status': 'success'})

# Función que predice el texto corregido
def predict(text):
    logger.info(text)
    inputs = tokenizer(f"corregir: {text}", return_tensors="pt", padding=False, truncation=False, max_length=270)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=270, num_beams=10, early_stopping=False)

    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(corrected_text)
    return corrected_text

# para obtener un segmento de subtítulo
@app.route('/get_subtitle_segment', methods=['POST'])
def get_subtitle_segment():
    global transcript_data
    # el fronten pasa el tiempo de inicio y final de los subtitulos
    start_time = request.json['start_time']
    end_time = request.json['end_time']
 
    # filtra los subtitulos para obtener el rango start-end y ejecuta el modelo con el texto para obtener la predicción que devolveremos al front
    segment = [entry for entry in transcript_data if entry['start'] >= start_time and entry['start'] < end_time]
    corrected_segment = [{'start': entry['start'], 'text': predict(entry['text'])} for entry in segment]
    return jsonify({
        'original_segment': segment,
        'corrected_segment': corrected_segment
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
