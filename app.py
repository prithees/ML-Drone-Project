from flask import Flask, request, render_template, jsonify
import os

import history
import matching
import spray_area

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB per file
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MIN_REFERENCE_PHOTOS = 3
MAX_REFERENCE_PHOTOS = 10

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _is_htmx():
    return request.headers.get('HX-Request') == 'true'


def _error(fragment_name, message, status_code=400):
    if _is_htmx():
        return render_template(fragment_name, error=message)
    return message, status_code


def _respond(fragment_name, full_page_name, **context):
    if _is_htmx():
        return render_template(fragment_name, **context)
    return render_template(full_page_name, **context)


def _validate_enroll(files, guidance):
    """Returns an error message, or None if the enroll request is valid."""
    if not files:
        return "Please choose at least one reference photo."
    if len(files) < MIN_REFERENCE_PHOTOS:
        return f"Please upload at least {MIN_REFERENCE_PHOTOS} reference photos."
    if len(files) > MAX_REFERENCE_PHOTOS:
        return f"Please upload at most {MAX_REFERENCE_PHOTOS} reference photos."
    if not guidance:
        return "Please describe what to do when this is matched."
    for f in files:
        if not allowed_file(f.filename):
            return f"Unsupported file type: {f.filename}. Use PNG or JPEG."
    return None


def _run_enroll(files, guidance):
    embeddings = []
    for f in files:
        filename = f"ref_{len(embeddings)}_{f.filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(file_path)
        embeddings.append(matching.embed_image(file_path))

    matching.save_gallery(embeddings, guidance)
    return len(embeddings)


def _run_scan(file):
    reference_embeddings, guidance = matching.load_gallery()
    if reference_embeddings is None:
        return None, None, None, "No target has been set yet. Set a target first."

    filename = "scan_query.png"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    query_embedding = matching.embed_image(file_path)
    score = matching.best_match_score(query_embedding, reference_embeddings)
    matched = matching.is_match(score)
    # sprayed=False: a scan through the web/app API is a preview, it doesn't
    # trigger the drone's GPIO — only spray_service.SpraySession does that.
    history.log_event(matched, score, sprayed=False, source="api")
    return matched, score, guidance, None


@app.route("/")
def home():
    embeddings, guidance = matching.load_gallery()
    target_set = embeddings is not None
    num_references = len(embeddings) if target_set else 0
    return render_template(
        'index.html',
        target_set=target_set,
        num_references=num_references,
        guidance=guidance,
    )


@app.route("/enroll", methods=['POST'])
def enroll():
    files = [f for f in request.files.getlist('file') if f.filename]
    guidance = request.form.get('guidance', '').strip()

    error = _validate_enroll(files, guidance)
    if error:
        return _error('_enroll_result_fragment.html', error)

    num_references = _run_enroll(files, guidance)

    return _respond(
        '_enroll_result_fragment.html',
        'enroll_result.html',
        num_references=num_references,
        guidance=guidance,
    )


@app.route("/scan", methods=['POST'])
def scan():
    if 'file' not in request.files or not request.files['file'].filename:
        return _error('_scan_result_fragment.html', "No file selected.")
    file = request.files['file']
    if not allowed_file(file.filename):
        return _error('_scan_result_fragment.html', "Unsupported file type. Please upload a PNG or JPEG image.")

    matched, score, guidance, error = _run_scan(file)
    if error:
        return _error('_scan_result_fragment.html', error)

    return _respond(
        '_scan_result_fragment.html',
        'scan_result.html',
        matched=matched,
        score=score,
        guidance=guidance,
    )


def _parse_area_points(raw_points):
    """Returns (points, error). points is a list of (lat, lng) tuples on success."""
    if not isinstance(raw_points, list) or len(raw_points) < spray_area.MIN_POLYGON_POINTS:
        return None, f"An area needs at least {spray_area.MIN_POLYGON_POINTS} points."
    points = []
    for p in raw_points:
        try:
            lat = float(p["lat"])
            lng = float(p["lng"])
        except (KeyError, TypeError, ValueError):
            return None, "Each point needs numeric 'lat' and 'lng'."
        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            return None, "Point out of range: lat must be -90..90, lng must be -180..180."
        points.append((lat, lng))
    return points, None


@app.route("/api/area", methods=['GET'])
def api_get_area():
    points = spray_area.load_area()
    return jsonify(
        area_set=points is not None,
        points=[{"lat": lat, "lng": lng} for lat, lng in points] if points else [],
    )


@app.route("/api/area", methods=['POST'])
def api_set_area():
    data = request.get_json(silent=True) or {}
    points, error = _parse_area_points(data.get('points'))
    if error:
        return jsonify(error=error), 400

    spray_area.save_area(points)
    return jsonify(points=[{"lat": lat, "lng": lng} for lat, lng in points])


@app.route("/api/history", methods=['GET'])
def api_history():
    limit = request.args.get('limit', type=int)
    return jsonify(entries=history.load_history(limit=limit))


@app.route("/api/threshold", methods=['GET'])
def api_get_threshold():
    return jsonify(threshold=matching.get_match_threshold())


@app.route("/api/threshold", methods=['POST'])
def api_set_threshold():
    data = request.get_json(silent=True) or {}
    try:
        threshold = float(data.get('threshold'))
    except (TypeError, ValueError):
        return jsonify(error="threshold must be a number."), 400
    if not (0 <= threshold <= 1):
        return jsonify(error="threshold must be between 0 and 1."), 400

    matching.set_match_threshold(threshold)
    return jsonify(threshold=threshold)


# --- Spray control over HTTP -------------------------------------------
# An alternative control surface to ble_server.py's BLE characteristics,
# for reaching the drone beyond Bluetooth's ~10-30m range (e.g. over a
# cellular modem on the Pi + a tunnel, instead of local WiFi/BLE) — same
# start/stop/status shape, just over HTTP so it works over any network the
# Pi has. Run EITHER this app's spray endpoints OR ble_server.py, not both:
# each independently owns the GPIO pin via its own SpraySession, and running
# both at once risks two loops driving the same pin.
_spray_session = None
_spray_status = "idle"


def _on_spray_status(status):
    global _spray_status
    _spray_status = status


def _get_spray_session():
    global _spray_session
    if _spray_session is None:
        import spray_service  # deferred: needs RPi.GPIO, only importable on the Pi
        _spray_session = spray_service.SpraySession(status_callback=_on_spray_status)
    return _spray_session


@app.route("/api/spray/start", methods=['POST'])
def api_spray_start():
    _get_spray_session().start()
    return jsonify(status=_spray_status)


@app.route("/api/spray/stop", methods=['POST'])
def api_spray_stop():
    _get_spray_session().stop()
    return jsonify(status=_spray_status)


@app.route("/api/spray/status", methods=['GET'])
def api_spray_status():
    return jsonify(status=_spray_status)


@app.route("/api/status", methods=['GET'])
def api_status():
    embeddings, guidance = matching.load_gallery()
    target_set = embeddings is not None
    return jsonify(
        target_set=target_set,
        num_references=len(embeddings) if target_set else 0,
        guidance=guidance,
    )


@app.route("/api/enroll", methods=['POST'])
def api_enroll():
    files = [f for f in request.files.getlist('file') if f.filename]
    guidance = request.form.get('guidance', '').strip()

    error = _validate_enroll(files, guidance)
    if error:
        return jsonify(error=error), 400

    num_references = _run_enroll(files, guidance)

    return jsonify(num_references=num_references, guidance=guidance)


@app.route("/api/scan", methods=['POST'])
def api_scan():
    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify(error="No file selected."), 400
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify(error="Unsupported file type. Please upload a PNG or JPEG image."), 400

    matched, score, guidance, error = _run_scan(file)
    if error:
        return jsonify(error=error), 400

    return jsonify(matched=matched, score=score, guidance=guidance)


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    # Bind on all interfaces so the mobile app can reach this from a phone on
    # the same network, not just from this machine.
    app.run(host="0.0.0.0", debug=debug_mode)
