from flask import Flask, render_template, request, send_from_directory, send_file, redirect, url_for
from utils import create_asr_engine
from werkzeug.utils import secure_filename
from io import BytesIO
import os
import math

app = Flask(__name__)

UPLOAD_FOLDER = "/app/uploads"
FILES_PER_PAGE = 5

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def get_files(page, sort, search):
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

    total_pages = max(1, math.ceil(len(files) / FILES_PER_PAGE))

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

    selected_model = request.form.get("selected_model") or "canary"
    device = request.form.get("device") or "cpu"
    strategy = request.form.get("strategy") or "beam"

    beam_size = request.form.get("beam_size", 5, type=int)
    batch_size = request.form.get("batch_size", 1, type=int)
    use_fp16 = request.form.get("fp16") == "on"

    len_pen = request.form.get("len_pen", 1.0, type=float)
    language = request.form.get("language") or "cs"
    return_hypotheses = request.form.get("return_hypotheses") == "on"

    alpha = request.form.get("alpha", 0.5, type=float)
    beta = request.form.get("beta", 1.0, type=float)

    whisper_language = request.form.get("whisper_language") or None
    temperature = request.form.get("temperature", 0.0, type=float)
    best_of = request.form.get("best_of", 5, type=int)
    vad_filter = request.form.get("vad_filter") == "on"

    action = request.form.get("action")

    if request.method == "POST" and action == "upload":
        file = request.files.get("file")

        if file and file.filename:
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))

        return redirect(url_for("index", page=1, sort=sort, search=search))

    if request.method == "POST" and action == "clear":
        return redirect(url_for("index", page=page, sort=sort, search=search))

    if request.method == "POST" and action == "use_file":
        filename = request.form.get("audio_file")

        if not filename:
            error = "No file selected"
        else:
            try:
                filename = secure_filename(filename)
                path = os.path.join(UPLOAD_FOLDER, filename)

                if not os.path.exists(path):
                    raise ValueError("File not found")

                engine = selected_model

                if engine == "canary":
                    asr = create_asr_engine(
                        engine,
                        device=device,
                        strategy=strategy,
                        beam_size=beam_size,
                        len_pen=len_pen,
                        language=language,
                        return_hypotheses=return_hypotheses,
                        batch_size=batch_size,
                        use_fp16=use_fp16
                    )

                elif engine == "parakeet":
                    asr = create_asr_engine(
                        engine,
                        device=device,
                        strategy=strategy,
                        beam_size=beam_size,
                        alpha=alpha,
                        beta=beta,
                        batch_size=batch_size,
                        use_fp16=use_fp16
                    )

                elif engine == "whisper":
                    asr = create_asr_engine(
                        engine,
                        device=device,
                        beam_size=beam_size,
                        language=whisper_language,
                        temperature=temperature,
                        vad_filter=vad_filter,
                        best_of=best_of
                    )

                else:
                    raise ValueError("Unknown model")

                transcript = asr.transcribe(path)

                if engine == "canary" and return_hypotheses:
                    transcript = transcript.text

            except Exception as e:
                error = f"Error: {e}"

    files, total_pages = get_files(page, sort, search)

    return render_template(
        "index.html",

        files=files,
        page=page,
        total_pages=total_pages,
        sort=sort,
        search=search,

        transcript=transcript,
        error=error,

        selected_model=selected_model,
        device=device,
        strategy=strategy,
        beam_size=beam_size,
        batch_size=batch_size,

        len_pen=len_pen,
        language=language,
        return_hypotheses=return_hypotheses,

        alpha=alpha,
        beta=beta,
        use_fp16=use_fp16,

        whisper_language=whisper_language,
        temperature=temperature,
        best_of=best_of,
        vad_filter=vad_filter
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
    
