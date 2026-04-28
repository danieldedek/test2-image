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


@app.route("/", methods=["GET", "POST"])
def index():
    transcript = None
    error = None

    page = int(request.args.get("page", 1))
    sort = request.args.get("sort", "date")
    search = request.args.get("search", "")

    selected_model = request.form.get("selected_model") or request.args.get("selected_model") or "canary"

    device = request.form.get("device") or "cpu"
    strategy = request.form.get("strategy") or "beam"
    beam_size = int(request.form.get("beam_size", 5))
    use_fp16 = request.form.get("fp16") == "on"

    len_pen = float(request.form.get("len_pen", 1.0))
    language = request.form.get("language") or "cs"
    return_hypotheses = request.form.get("return_hypotheses") == "on"
    batch_size = int(request.form.get("batch_size", 1))

    alpha = float(request.form.get("alpha", 0.5))
    beta = float(request.form.get("beta", 1.0))

    if request.method == "POST":

        action = request.form.get("action")

        if action == "select_model":
            return redirect(url_for(
                "index",
                selected_model=selected_model,
                page=page,
                sort=sort,
                search=search
            ))

        if action == "upload" and "file" in request.files:
            file = request.files["file"]

            if file and file.filename:
                filename = secure_filename(file.filename)
                file.save(os.path.join(UPLOAD_FOLDER, filename))

            return redirect(url_for("index", selected_model=selected_model))

        if action == "use_file":

            name = request.form.get("audio_file")
            wav_path = os.path.join(UPLOAD_FOLDER, secure_filename(name))

            if not os.path.exists(wav_path):
                error = "File not found"
            else:
                try:

                    if selected_model == "parakeet":
                        asr = create_asr_engine(
                            "parakeet",
                            device=device,
                            strategy=strategy,
                            beam_size=beam_size,
                            alpha=alpha,
                            beta=beta,
                            batch_size=batch_size,
                            use_fp16=use_fp16
                        )

                    else:
                        asr = create_asr_engine(
                            "canary",
                            device=device,
                            strategy=strategy,
                            beam_size=beam_size,
                            len_pen=len_pen,
                            language=language,
                            return_hypotheses=return_hypotheses,
                            batch_size=batch_size,
                            use_fp16=use_fp16
                        )

                    result = asr.transcribe(wav_path)
                    transcript = result

                    if selected_model == "canary" and return_hypotheses:
                        try:
                            transcript = result.text
                        except:
                            transcript = str(result)

                except Exception as e:
                    error = str(e)

    files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".wav")]

    if search:
        files = [f for f in files if search.lower() in f.lower()]

    if sort == "name":
        files.sort()
    else:
        files.sort(key=lambda f: os.path.getmtime(os.path.join(UPLOAD_FOLDER, f)), reverse=True)

    total_pages = max(1, math.ceil(len(files) / FILES_PER_PAGE))
    files = files[(page - 1) * FILES_PER_PAGE: page * FILES_PER_PAGE]

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

        len_pen=len_pen,
        language=language,
        return_hypotheses=return_hypotheses,
        batch_size=batch_size,

        alpha=alpha,
        beta=beta,

        use_fp16=use_fp16
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
    
