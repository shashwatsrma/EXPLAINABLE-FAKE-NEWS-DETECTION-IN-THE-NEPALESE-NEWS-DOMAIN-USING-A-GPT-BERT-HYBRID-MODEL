"""# ============================================================
# APP.PY — Flask Web Application
# ============================================================

from flask import Flask, render_template, request, jsonify
from predict import FakeNewsPredictor
from config import DEVICE

app = Flask(__name__)

# Load model at startup
predictor = FakeNewsPredictor()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "No text provided."}), 400

        text = data["text"].strip()
        if len(text) < 10:
            return jsonify({"error": "Text too short. Please provide a longer article."}), 400

        include_explanation = data.get("explain", True)
        result = predictor.predict(text, explain=include_explanation)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE), "model": "GBERT Fusion"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
"""

# ============================================================
# APP.PY — Flask Web Application
# ============================================================

from flask import Flask, render_template, request, jsonify
from predict import FakeNewsPredictor
from config import DEVICE

app = Flask(__name__)

# ── CORS: allow frontend JS to call the API from any origin ──
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/api/predict", methods=["OPTIONS"])
def predict_options():
    """Handle CORS preflight for the predict endpoint."""
    return "", 204


# Load model once at startup
print("Loading model at startup...")
predictor = FakeNewsPredictor()
print("Model ready.")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json(force=True, silent=True)

        if not data or "text" not in data:
            return jsonify({"error": "No text provided."}), 400

        text = data["text"].strip()
        if len(text) < 10:
            return jsonify({"error": "Text too short. Please provide a longer article."}), 400

        include_explanation = data.get("explain", True)
        result = predictor.predict(text, explain=include_explanation)

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()          # prints full error to terminal
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE), "model": "GBERT Fusion"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)