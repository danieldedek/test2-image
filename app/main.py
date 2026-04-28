from flask import Flask, render_template, request, send_from_directory, send_file, redirect, url_for
from utils import create_asr_engine
from io import BytesIO
from werkzeug.utils import secure_filename
import os
import math

app = Flask(__name__)

UPLOAD_FOLDER = "/app/uploads"
FILES_PER_PAGE = 5

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def get_wav_files(page, sort, search):
    files = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(".wav")]

    if search:
        files = [f for f in files if search.lower() in f.lower()]

    if sort == "name":
        files.sort(key=str.lower)
    else:
        files.sort(
            key=lambda f: os.path.getmtime(os.path.join(UPLOAD_FOLDER, f)),
            reverse=True
        )

    total = len(files)
    total_pages = max(1, math.ceil(total / FILES_PER_PAGE))

    start = (page - 1) * FILES_PER_PAGE
    end = start + FILES_PER_PAGE

    return files[start:end], total_pages


@app.route("/", methods=["GET", "POST"])
def index():
    transcript = None
    error = None

    page = int(request.args.get("page", 1))
    sort = request.args.get("sort", "date")
    search = request.args.get("search", "")

    engine = request.form.get("engine") or "canary"
    device = request.form.get("device") or "cpu"

    if request.method == "POST":

        action = request.form.get("action")

        if action == "clear":
            return redirect(url_for("index", page=page, sort=sort, search=search))

        file = request.files.get("file")
        if file and file.filename:
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            return redirect(url_for("index", page=1, sort=sort, search=search))

        audio_file = request.form.get("audio_file")
        if audio_file:
            try:
                name = secure_filename(audio_file)
                wav_path = os.path.join(UPLOAD_FOLDER, name)

                if not os.path.exists(wav_path):
                    raise ValueError("File does not exist")

                asr = create_asr_engine(engine, device=device)
                asr.download()

                result = asr.transcribe(wav_path)

                transcript = result.text if hasattr(result, "text") else result

            except Exception as e:
                error = f"Error: {e}"

    files, total_pages = get_wav_files(page, sort, search)

    return render_template(
        "index.html",
        transcript=transcript,
        error=error,
        files=files,
        page=page,
        total_pages=total_pages,
        sort=sort,
        search=search,
        engine=engine,
        device=device
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/download", methods=["POST"])
def download():
    buffer = BytesIO(request.form.get("transcript", "").encode("utf-8"))
    return send_file(buffer, as_attachment=True, download_name="transcript.txt")


@app.route("/delete/<filename>", methods=["POST"])
def delete_file(filename):
    filename = secure_filename(filename)
    path = os.path.join(UPLOAD_FOLDER, filename)

    if os.path.exists(path):
        os.remove(path)

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    
