from flask import Flask, render_template, request, send_file, send_from_directory
from utils import create_asr_engine
from io import BytesIO
from werkzeug.utils import secure_filename
import os
import subprocess
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = "/app/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def convert_webm_to_wav(webm_path, wav_path):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", webm_path,
            "-ac", "1",
            "-ar", "16000",
            wav_path
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


@app.route("/", methods=["GET", "POST"])
def index():
    transcript = None
    error = None
    audio_filename = None

    if request.method == "POST":
        if request.form.get("action") == "clear":
            return render_template("index.html")

        engine_name = request.form.get("engine")

        try:
            if "audio_blob" in request.files and request.files["audio_blob"].filename:
                webm = request.files["audio_blob"]

                custom_name = secure_filename(request.form.get("recording_name", "").strip())

                if custom_name:
                    base_name = custom_name
                else:
                    base_name = uuid.uuid4().hex

                webm_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{base_name}.webm")
                wav_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{base_name}.wav")

                webm.save(webm_path)
                convert_webm_to_wav(webm_path, wav_path)

                audio_file = wav_path
                audio_filename = os.path.basename(wav_path)

            else:
                name = secure_filename(request.form.get("audio_file", "").strip())

                if not name:
                    raise ValueError(f"No file selected. Upload folder: {app.config['UPLOAD_FOLDER']}")

                wav_path = os.path.join(app.config["UPLOAD_FOLDER"], name)

                if not os.path.exists(wav_path):
                    raise ValueError("File does not exist")

                audio_file = wav_path
                audio_filename = name

            asr = create_asr_engine(engine_name)
            asr.download()
            transcript = asr.transcribe(audio_file)

            if engine_name in ("canary", "parakeet"):
                transcript = transcript.text

        except Exception as e:
            error = f"Error: {e}"

    return render_template(
        "index.html",
        transcript=transcript,
        error=error,
        audio_filename=audio_filename
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/download", methods=["POST"])
def download():
    text = request.form.get("transcript", "")
    buffer = BytesIO(text.encode("utf-8"))
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name="transcript.txt",
        mimetype="text/plain"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
