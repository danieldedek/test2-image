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

    engine = request.form.get("engine") or "canary"
    device = request.form.get("device") or "cpu"

    strategy = request.form.get("strategy") or "beam"
    beam_size = int(request.form.get("beam_size", 5))

    # CANARY
    len_pen = float(request.form.get("len_pen", 1.0))
    language = request.form.get("language") or "cs"
    return_hypotheses = request.form.get("return_hypotheses") == "on"

    # PARAKEET
    alpha = float(request.form.get("alpha", 0.5))
    beta = float(request.form.get("beta", 1.0))

    use_fp16 = request.form.get("fp16") == "on"

    if request.method == "POST" and "audio_file" in request.form:

        filename = secure_filename(request.form.get("audio_file"))
        path = os.path.join(UPLOAD_FOLDER, filename)

        if engine == "parakeet":
            asr = create_asr_engine(
                engine,
                device=device,
                strategy=strategy,
                beam_size=beam_size,
                alpha=alpha,
                beta=beta,
                batch_size=4,
                use_fp16=use_fp16
            )

        elif engine == "canary":
            asr = create_asr_engine(
                engine,
                device=device,
                strategy=strategy,
                beam_size=beam_size,
                len_pen=len_pen,
                language=language,
                return_hypotheses=return_hypotheses,
                use_fp16=use_fp16
            )

        transcript = asr.transcribe(path)

        if return_hypotheses and engine == "canary":
            transcript = transcript.text

    return render_template(
        "index.html",
        engine=engine,
        device=device,
        strategy=strategy,
        beam_size=beam_size,
        len_pen=len_pen,
        language=language,
        alpha=alpha,
        beta=beta,
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
    
