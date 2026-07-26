# AgroScan (mobile)

Flutter client for the AgroScan few-shot image matching backend (`../app.py`).
Targets Android and iOS. Lets you set a target from reference photos and scan
new photos against it, from a phone instead of a browser.

## Setup

```
flutter pub get
```

## Running against the Flask backend

Start the Flask server from the repo root first:

```
python app.py
```

Then run the app:

```
flutter run
```

By default the app points at:
- `http://10.0.2.2:5000` on the Android emulator (routes to the host machine's localhost)
- `http://localhost:5000` everywhere else (iOS simulator)

For a physical device, open the app's Settings screen (gear icon, top right)
and set the server URL to your computer's LAN IP, e.g. `http://192.168.1.20:5000`
— the phone and the computer running Flask need to be on the same network.
Flask's dev server only listens on localhost by default; run it with
`flask run --host=0.0.0.0` (or `app.run(host="0.0.0.0")`) to accept connections
from other devices on the network.

## Structure

- `lib/api_client.dart` — talks to `/api/status`, `/api/enroll`, `/api/scan` on the Flask backend.
- `lib/screens/home_screen.dart` — current target status + navigation.
- `lib/screens/enroll_screen.dart` — pick 3-10 reference photos + guidance text, set the target.
- `lib/screens/scan_screen.dart` — pick/take a photo, scan it against the target.
- `lib/screens/settings_screen.dart` — configure and test the backend server URL.
