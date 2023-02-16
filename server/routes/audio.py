import os
from flask import Blueprint, current_app, request

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
