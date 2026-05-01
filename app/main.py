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


def get_int(name, default):
    value = request.form.get(name)
    return int(value) if value not in (None, "") else default


def get_float(name, default):
    value = request.form.get(name)
    return float(value) if value not in (None, "") else default


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
    whisper_segments = []

    engine = request.args.get("model", "canary")

    page = int(request.args.get("page", 1))
    sort = request.args.get("sort", "date")
    search = request.args.get("search", "")

    if request.method == "POST":
        engine = request.form.get("selected_model") or engine

    device = request.form.get("device") or "cpu"
    strategy = request.form.get("strategy") or "beam"

    beam_size = get_int("beam_size", 5)
    use_fp16 = request.form.get("fp16") == "on"

    len_pen = get_float("len_pen", 1.0)
    language = request.form.get("language") or "en"
    return_hypotheses = request.form.get("return_hypotheses") == "on"

    whisper_language = request.form.get("whisper_language") or "en"
    temperature = get_float("temperature", 0.0)
    best_of = get_int("best_of", 5)
    vad_filter = request.form.get("vad_filter") == "on"
    model_size = request.form.get("model_size") or "medium"
    condition_on_previous_text = request.form.get("condition_on_previous_text") == "on"
    show_timestamps = request.form.get("show_timestamps") == "on"
    show_confidence = request.form.get("show_confidence") == "on"
    

    action = request.form.get("action")

    if request.method == "POST" and action == "upload":
        file = request.files.get("file")
        if file and file.filename:
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))

        return redirect(url_for("index", model=engine, page=page, sort=sort, search=search))

    if request.method == "POST" and action == "select_model":
        selected = request.form.get("selected_model") or "canary"
        return redirect(url_for("index", model=selected, page=page, sort=sort, search=search))

    if request.method == "POST" and action == "clear":
        return redirect(url_for("index", model=engine, page=page, sort=sort, search=search))

    if request.method == "POST" and action == "use_file":
        filename = request.form.get("audio_file")

        if not filename:
            error = "No file selected"
        else:
            try:
                path = os.path.join(UPLOAD_FOLDER, secure_filename(filename))

                if engine == "canary":
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

                elif engine == "parakeet":
                    asr = create_asr_engine(
                        engine,
                        device=device,
                        strategy=strategy,
                        beam_size=beam_size,
                        use_fp16=use_fp16,
                        return_hypotheses=return_hypotheses
                    )

                    transcript = asr.transcribe(path)

                elif engine == "whisper":
                    asr = create_asr_engine(
                        engine,
                        model_size=model_size,
                        device=device,
                        beam_size=beam_size,
                        language=whisper_language,
                        temperature=temperature,
                        vad_filter=vad_filter,
                        best_of=best_of,
                        condition_on_previous_text=condition_on_previous_text
                    )

                    if show_timestamps or show_confidence:
                        segments = asr.transcribe(path, return_segments=True)
                        transcript = " ".join(seg.text for seg in segments)
                        whisper_segments = [
                            {
                                "start": round(seg.start, 2),
                                "end": round(seg.end, 2),
                                "text": seg.text.strip(),
                                "confidence": round(1 - seg.no_speech_prob, 2),
                                "words": [
                                    {
                                        "word": w.word,
                                        "start": round(w.start, 2),
                                        "end": round(w.end, 2),
                                        "confidence": round(w.probability, 2)
                                    }
                                    for w in (seg.words or [])
                                ]
                            }
                            for seg in segments
                        ]
                    else:
                        transcript = asr.transcribe(path)
                        whisper_segments = []

            except Exception as e:
                error = f"Error: {e}"

    files, total_pages = get_wav_files(page, sort, search)

    return render_template(
        "index.html",
        files=files,
        transcript=transcript,
        error=error,
        page=page,
        total_pages=total_pages,
        sort=sort,
        search=search,
        selected_model=engine,
        device=device,
        strategy=strategy,
        beam_size=beam_size,
        len_pen=len_pen,
        language=language,
        return_hypotheses=return_hypotheses,
        use_fp16=use_fp16,
        whisper_language=whisper_language,
        temperature=temperature,
        best_of=best_of,
        vad_filter=vad_filter,
        model_size=model_size,
        condition_on_previous_text=condition_on_previous_text,
        show_timestamps=show_timestamps,
        show_confidence=show_confidence,
        whisper_segments=whisper_segments
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
    
