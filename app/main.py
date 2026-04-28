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


@app.route("/", methods=["GET", "POST"])
def index():
    transcript = None
    error = None

    page = int(request.args.get("page", 1))
    sort = request.args.get("sort", "date")
    search = request.args.get("search", "")

    # DEFAULTS
    engine = request.form.get("engine") or "canary"
    device = request.form.get("device") or "cpu"

    strategy = request.form.get("strategy") or "beam"
    beam_size = int(request.form.get("beam_size", 5))
    len_pen = float(request.form.get("len_pen", 1.0))
    batch_size = int(request.form.get("batch_size", 1))
    language = request.form.get("language") or "cs"
    task = request.form.get("task") or "transcribe"

    use_fp16 = request.form.get("fp16") == "on"
    return_hypotheses = request.form.get("hypotheses") == "on"

    if request.method == "POST":

        if request.form.get("action") == "clear":
            return redirect(url_for("index", page=page, sort=sort, search=search))

        # upload
        if "file" in request.files:
            file = request.files["file"]
            if file and file.filename:
                filename = secure_filename(file.filename)
                file.save(os.path.join(UPLOAD_FOLDER, filename))
                return redirect(url_for("index", page=1, sort=sort, search=search))

        # transcribe
        if "audio_file" in request.form:
            try:
                name = secure_filename(request.form.get("audio_file"))
                wav_path = os.path.join(UPLOAD_FOLDER, name)

                if not os.path.exists(wav_path):
                    raise ValueError("File does not exist")

                asr = create_asr_engine(
                    engine,
                    device=device,
                    strategy=strategy,
                    beam_size=beam_size,
                    len_pen=len_pen,
                    batch_size=batch_size,
                    language=language,
                    task=task,
                    use_fp16=use_fp16,
                    return_hypotheses=return_hypotheses
                )
                asr.download()
                transcript = asr.transcribe(wav_path)

                if return_hypotheses:
                    transcript = transcript.text

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
        device=device,
        strategy=strategy,
        beam_size=beam_size,
        len_pen=len_pen,
        batch_size=batch_size,
        language=language,
        task=task,
        use_fp16=use_fp16,
        return_hypotheses=return_hypotheses
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
