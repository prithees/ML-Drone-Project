# ML-Drone-Project

Few-shot image matching for drone pesticide spraying: enroll a few reference
photos of what to spray and a Flask backend matches new photos against them
by embedding similarity (MobileNetV2), no fixed class list or training.

- `app.py`, `matching.py`, `templates/`, `static/` — the Flask + htmx web app.
- `testing.py` — drone-side scan loop (Raspberry Pi).
- `mobile/` — Flutter app (Android + iOS) for enrolling/scanning from a phone against the same Flask backend. See `mobile/README.md`.
