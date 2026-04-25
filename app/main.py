from flask import Flask, render_template, request, send_from_directory, send_file, redirect, url_for
from utils import create_asr_engine
from io import BytesIO
from werkzeug.utils import secure_filename
import os
import math

app = Flask(__name__)

UPLOAD_FOLDER = "/app/uploads"
FILES_PER_PAGE = 10

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def get_wav_files(page, sort):
    all_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(".wav")]

    if sort == "name":
        all_files.sort()
    else:
        all_files.sort(key=lambda f: os.path.getmtime(os.path.join(UPLOAD_FOLDER, f)), reverse=True)

    total = len(all_files)
    start = (page - 1) * FILES_PER_PAGE
    end = start + FILES_PER_PAGE

    files = all_files[start:end]

    has_next = end < total
    has_prev = start > 0
    total_pages = max(1, math.ceil(total / FILES_PER_PAGE))

    return files, has_next, has_prev, total_pages


@app.route("/", methods=["GET", "POST"])
def index():
    transcript = None
    error = None

    page = int(request.args.get("page", 1))
    sort = request.args.get("sort", "date")

    if request.method == "POST":
        if request.form.get("action") == "clear":
            return redirect(url_for("index", sort=sort, page=page))

        if "file" in request.files:
            file = request.files["file"]
            if file and file.filename != "":
                filename = secure_filename(file.filename)
                path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(path)
                return redirect(url_for("index", page=1, sort=sort))

        if "audio_file" in request.form:
            engine_name = request.form.get("engine")
            device = request.form.get("device", "cpu")

            try:
                name = secure_filename(request.form.get("audio_file"))
                wav_path = os.path.join(app.config["UPLOAD_FOLDER"], name)

                if not os.path.exists(wav_path):
                    raise ValueError("File does not exist")

                asr = create_asr_engine(engine_name, device=device)
                asr.download()
                transcript = asr.transcribe(wav_path)

                if engine_name in ("canary", "parakeet"):
                    transcript = transcript.text

            except Exception as e:
                error = f"Error: {e}"

    files, has_next, has_prev, total_pages = get_wav_files(page, sort)

    return render_template(
        "index.html",
        transcript=transcript,
        error=error,
        files=files,
        page=page,
        has_next=has_next,
        has_prev=has_prev,
        total_pages=total_pages,
        sort=sort
    )


@app.route("/uploads/<filename>")
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


@app.route("/delete/<filename>", methods=["POST"])
def delete_file(filename):
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if os.path.exists(path):
        os.remove(path)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
