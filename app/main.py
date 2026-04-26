from flask import Flask, render_template, request, send_from_directory, send_file, redirect, url_for
from utils import create_asr_engine
from io import BytesIO
from werkzeug.utils import secure_filename
import os
import math
import threading
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "/app/uploads"
FILES_PER_PAGE = 5

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

tasks = {}


def get_wav_files(page, sort, search):
    files = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(".wav")]

    if search:
        files = [f for f in files if search.lower() in f.lower()]

    if sort == "name":
        files.sort()
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


def run_asr(task_id, wav_path, engine, device):
    try:
        asr = create_asr_engine(engine, device=device)
        asr.download()
        result = asr.transcribe(wav_path)

        if engine in ("canary", "parakeet"):
            result = result.text

        tasks[task_id] = {"status": "done", "result": result}

    except Exception as e:
        tasks[task_id] = {"status": "error", "error": str(e)}


@app.route("/", methods=["GET", "POST"])
def index():
    transcript = None
    error = None

    page = int(request.args.get("page", 1))
    sort = request.args.get("sort", "date")
    search = request.args.get("search", "")
    task_id = request.args.get("task")

    engine = request.form.get("engine") or "canary"
    device = request.form.get("device") or "cpu"

    if request.method == "POST":

        if request.form.get("action") == "clear":
            return redirect(url_for("index", page=page, sort=sort, search=search))

        if "file" in request.files:
            file = request.files["file"]
            if file and file.filename:
                filename = secure_filename(file.filename)
                file.save(os.path.join(UPLOAD_FOLDER, filename))
                return redirect(url_for("index", page=1, sort=sort, search=search))

        if "audio_file" in request.form:
            name = secure_filename(request.form.get("audio_file"))
            wav_path = os.path.join(UPLOAD_FOLDER, name)

            task_id = str(uuid.uuid4())
            tasks[task_id] = {"status": "running"}

            thread = threading.Thread(
                target=run_asr,
                args=(task_id, wav_path, engine, device)
            )
            thread.start()

            return redirect(url_for("index", task=task_id, page=page, sort=sort, search=search))

    if task_id and task_id in tasks:
        task = tasks[task_id]

        if task["status"] == "done":
            transcript = task["result"]

        elif task["status"] == "error":
            error = task["error"]

        else:
            error = "Processing..."

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
        device=device,
        task_id=task_id
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
    path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(path):
        os.remove(path)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    
