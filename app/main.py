from flask import Flask, render_template, request, send_from_directory, send_file, redirect, url_for
from utils import create_asr_engine
from werkzeug.utils import secure_filename
from io import BytesIO
import os

app = Flask(__name__)

UPLOAD_FOLDER = "/app/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def get_int(name, default):
    value = request.form.get(name)
    return int(value) if value not in (None, "") else default


def get_float(name, default):
    value = request.form.get(name)
    return float(value) if value not in (None, "") else default


def get_files():
    return [
        f for f in os.listdir(UPLOAD_FOLDER)
        if f.lower().endswith(".wav")
    ]


@app.route("/", methods=["GET", "POST"])
def index():
    transcript = None
    error = None

    selected_model = request.form.get("selected_model") or "canary"
    engine = selected_model

    device = request.form.get("device") or "cpu"
    strategy = request.form.get("strategy") or "beam"

    beam_size = get_int("beam_size", 5)
    batch_size = get_int("batch_size", 1)
    use_fp16 = request.form.get("fp16") == "on"

    len_pen = get_float("len_pen", 1.0)
    language = request.form.get("language") or "cs"
    return_hypotheses = request.form.get("return_hypotheses") == "on"

    alpha = get_float("alpha", 0.5)
    beta = get_float("beta", 1.0)

    whisper_language = request.form.get("whisper_language") or None
    temperature = get_float("temperature", 0.0)
    best_of = get_int("best_of", 5)
    vad_filter = request.form.get("vad_filter") == "on"

    action = request.form.get("action")

    if request.method == "POST" and action == "upload":
        file = request.files.get("file")

        if file and file.filename:
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))

        return redirect(url_for("index"))

    if request.method == "POST" and action == "select_model":
        return render_template(
            "index.html",

            files=get_files(),
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
            vad_filter=vad_filter,

            transcript=None,
            error=None
        )

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

    return render_template(
        "index.html",

        files=get_files(),
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
    path = os.path.join(UPLOAD_FOLDER, filename)

    if os.path.exists(path):
        os.remove(path)

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
