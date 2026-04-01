from flask import Flask, render_template, request, send_from_directory, send_file
from utils import create_asr_engine
from io import BytesIO
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    transcript = None
    error = None
    audio_filename = None
    segments = None

    if request.method == "POST":
        if request.form.get("action") == "clear":
            return render_template("index.html")

        engine_name = request.form.get("engine")
        language = request.form.get("language")
        device = request.form.get("device", "cpu")
        timestamps = "timestamps" in request.form
        confidence = "confidence" in request.form
        verbose = "verbose" in request.form

        try:
            name = secure_filename(request.form.get("audio_file", "").strip())
            if not name:
                raise ValueError("No file selected")

            wav_path = os.path.join(app.config["UPLOAD_FOLDER"], name)
            if not os.path.exists(wav_path):
                raise ValueError("File does not exist")

            audio_file = wav_path
            audio_filename = name

            asr = create_asr_engine(engine_name, device=device)
            asr.download()
            result = asr.transcribe(
                audio_file,
                language=language,
                timestamps=timestamps,
                confidence=confidence,
                verbose=verbose
            )

            if isinstance(result, dict):
                transcript = result.get("text")
                segments = result.get("segments")
            else:
                transcript = result

        except Exception as e:
            error = f"Error: {e}"

    return render_template(
        "index.html",
        transcript=transcript,
        segments=segments,
        audio_filename=audio_filename,
        error=error
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(file_path):
        return f"File not found: {file_path}", 404

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename,
        mimetype="audio/wav"
    )


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
