import os
from flask import Blueprint, current_app, request
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

    return "TODO"


@bp.route('/audio/<string:uuid>', methods=["GET"])
def predict(uuid):
    path = current_app.config['UPLOAD_DIR']+'/'+uuid+'.wav'
    return SpeechToText.predict(path)


@bp.route('/audio/<string:uuid>', methods=["DELETE"])
def remove_audio(uuid):
    path = current_app.config['UPLOAD_DIR']+'/'+uuid+'.wav'
    if os.path.exists(path):
        os.remove(path)
        return f"File deleted:{path}"
    else:
        return "The file does not exist"
