from flask import Flask, render_template, request, send_from_directory, send_file, redirect, url_for, session
from utils import create_asr_engine
from io import BytesIO
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.secret_key = "dev"

UPLOAD_FOLDER = "/app/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def get_wav_files():
    return [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(".wav")]


def get_file_info(files):
    return [
        {
            "name": f,
            "mtime": os.path.getmtime(os.path.join(UPLOAD_FOLDER, f))
        }
        for f in files
    ]


@app.route("/", methods=["GET", "POST"])
def index():
    transcript = None
    error = None

    selected_file = session.get("selected_file")

    if request.method == "POST":
        if request.form.get("action") == "clear":
            session.pop("selected_file", None)
            return redirect(url_for("index"))

        if "file" in request.files:
            file = request.files["file"]
            if file and file.filename:
                filename = secure_filename(file.filename)
                path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(path)
                return redirect(url_for("index"))

        if "audio_file" in request.form:
            engine_name = request.form.get("engine")
            device = request.form.get("device", "cpu")

            try:
                name = secure_filename(request.form.get("audio_file"))
                wav_path = os.path.join(app.config["UPLOAD_FOLDER"], name)

                if not os.path.exists(wav_path):
                    raise ValueError("File does not exist")

                session["selected_file"] = name
                selected_file = name

                asr = create_asr_engine(engine_name, device=device)
                asr.download()
                transcript = asr.transcribe(wav_path)

                if engine_name in ("canary", "parakeet"):
                    transcript = transcript.text

            except Exception as e:
                error = f"Error: {e}"

    search = request.args.get("q", "").lower()
    sort = request.args.get("sort", "date")
    page = int(request.args.get("page", 1))
    per_page = 10

    files = get_wav_files()

    if search:
        files = [f for f in files if search in f.lower()]

    file_data = get_file_info(files)

    if sort == "alpha":
        file_data.sort(key=lambda x: x["name"].lower())
    else:
        file_data.sort(key=lambda x: x["mtime"], reverse=True)

    total = len(file_data)
    start = (page - 1) * per_page
    end = start + per_page
    paginated = file_data[start:end]

    return render_template(
        "index.html",
        transcript=transcript,
        error=error,
        files=paginated,
        page=page,
        total_pages=(total + per_page - 1) // per_page,
        search=search,
        sort=sort,
        selected_file=selected_file
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/download", methods=["POST"])
def download():
    text = request.form.get("transcript", "")
    buffer = BytesIO(text.encode("utf-8"))
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="transcript.txt", mimetype="text/plain")


@app.route("/delete/<filename>", methods=["POST"])
def delete_file(filename):
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if os.path.exists(path):
        os.remove(path)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
