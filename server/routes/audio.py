import codecs
import os

from flask import (Blueprint, current_app, json, request, send_file,
                   send_from_directory)

from server.utils.audio import SpeechToText

bp = Blueprint('audio', __name__)


@bp.route('/audio', methods=["POST", "GET"])
# TODO
# @token_required
def upload():
    if request.method == 'POST':
        upload_dir = current_app.config['UPLOAD_DIR']
        print(f'audio: {upload_dir}')
        for file in request.files.getlist('file'):
            file.save(os.path.join(upload_dir, file.filename))
        return "FILES UPLOADED"
    else:
        path = current_app.config['UPLOAD_DIR']
        data = []
        for p in os.listdir(path):
            with codecs.open(f'{path}/{p}', 'r', encoding='utf-8',
                             errors='ignore') as fdata:
                audio_blob = fdata.read()
                key = p.split('.')[0]
                word, index = SpeechToText.predict(f'{path}/{p}')

                data.append(
                    {'key': key, 'audio': audio_blob, 'predicted': word})
        response = current_app.response_class(
            response=json.dumps(data), status=200, mimetype='application/json'
        )
        return response


@bp.route('/audio/predict/<string:uuid>', methods=["GET"])
def predict(uuid):
    print(f'audio: {uuid}')
    path = current_app.config['UPLOAD_DIR']+'/'+uuid+'.wav'
    word, index = SpeechToText.predict(path)
    response = current_app.response_class(
        response=json.dumps({'class': word, 'index': str(index)}), status=200, mimetype='application/json'
    )
    return response

# FIXME


@bp.route('/audio/predicts/<string:uuid>', methods=["GET"])
def predicts(uuid):
    print(f'audio: {uuid}')
    path = current_app.config['UPLOAD_DIR']+'/'+uuid+'.wav'
    res = SpeechToText.predict_sentence(path)
    response = current_app.response_class(
        response=json.dumps({'ndarray': res.tolist()}), status=200, mimetype='application/json'
    )
    return response


@bp.route('/audio/<string:filename>', methods=["GET"])
def get_f(filename):
    filename = f'{filename}'
    uploads = os.path.join(current_app.root_path,
                           'audio')
    print(f'audio: {filename}')
    print(f'audio: {uploads}')
    return send_from_directory(uploads, filename)


@bp.route('/audio/<string:uuid>', methods=["DELETE"])
def remove_audio(uuid):
    path = current_app.config['UPLOAD_DIR']+'/'+uuid+'.wav'
    if os.path.exists(path):
        os.remove(path)
        response = current_app.response_class(
            response=json.dumps({'uuid': uuid}), status=202, mimetype='application/json'
        )
        return response
    else:
        return "The file does not exist"
