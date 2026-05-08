"""
SecureNet IDS — Flask Backend API
Serves both Random Forest and Decision Tree with model switching + metrics.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)

# ── Load all artifacts ────────────────────────────────────────────────────────
try:
    models = {
        'random_forest': joblib.load(os.path.join(BASE_DIR, 'model.pkl')),
        'decision_tree': joblib.load(os.path.join(BASE_DIR, 'dt_model.pkl'))
    }
    scaler        = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join(BASE_DIR, 'label_encoder.pkl'))
    feature_names = joblib.load(os.path.join(BASE_DIR, 'feature_names.pkl'))
    metrics       = joblib.load(os.path.join(BASE_DIR, 'metrics.pkl'))
    print(f'✅ Both models loaded — {len(feature_names)} features expected.')
except FileNotFoundError as e:
    print(f'❌ Model file not found: {e}')
    print('   Run the Jupyter Notebook or model_trainer.py first.')
    models = {}
    scaler = label_encoder = feature_names = metrics = None

active_model_key = 'random_forest'


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'SecureNet IDS API',
        'status': 'online',
        'models_loaded': list(models.keys()),
        'active_model': active_model_key
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'models_ready': len(models) == 2,
        'active_model': active_model_key
    })


@app.route('/features', methods=['GET'])
def features():
    if feature_names is None:
        return jsonify({'error': 'Model not loaded'}), 503
    return jsonify({'features': feature_names, 'count': len(feature_names)})


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Return accuracy, precision, recall, F1 for both models."""
    if metrics is None:
        return jsonify({'error': 'Metrics not loaded'}), 503
    return jsonify({'active_model': active_model_key, 'metrics': metrics})


@app.route('/set_model', methods=['POST'])
def set_model():
    """Switch the active model."""
    global active_model_key
    data = request.get_json(silent=True)
    if not data or 'model' not in data:
        return jsonify({'error': 'Provide {"model": "random_forest" or "decision_tree"}'}), 400

    key = data['model']
    if key not in models:
        return jsonify({'error': f'Unknown model: {key}'}), 400

    active_model_key = key
    return jsonify({
        'message': f'Switched to {key}',
        'active_model': active_model_key,
        'metrics': metrics.get(key, {}) if metrics else {}
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Classify a single network flow."""
    if not models:
        return jsonify({'error': 'Models not loaded. Run the notebook first.'}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'No JSON body provided'}), 400

    try:
        model        = models[active_model_key]
        values       = [float(data.get(feat, 0)) for feat in feature_names]
        X            = np.array(values).reshape(1, -1)
        X_scaled     = scaler.transform(X)
        prediction   = model.predict(X_scaled)[0]
        probs        = model.predict_proba(X_scaled)[0]
        label        = label_encoder.inverse_transform([prediction])[0]
        classes      = label_encoder.classes_.tolist()

        return jsonify({
            'prediction':    label,
            'is_attack':     label != 'BENIGN',
            'confidence':    round(float(max(probs)), 4),
            'probabilities': {c: round(float(p), 4) for c, p in zip(classes, probs)},
            'model_used':    active_model_key
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Classify multiple network flows."""
    if not models:
        return jsonify({'error': 'Models not loaded'}), 503

    data = request.get_json(silent=True)
    if not isinstance(data, list):
        return jsonify({'error': 'Expected a JSON array of records'}), 400

    model, results = models[active_model_key], []
    for record in data:
        try:
            values   = [float(record.get(feat, 0)) for feat in feature_names]
            X_scaled = scaler.transform(np.array(values).reshape(1, -1))
            pred     = model.predict(X_scaled)[0]
            probs    = model.predict_proba(X_scaled)[0]
            label    = label_encoder.inverse_transform([pred])[0]
            classes  = label_encoder.classes_.tolist()
            results.append({
                'prediction':    label,
                'is_attack':     label != 'BENIGN',
                'confidence':    round(float(max(probs)), 4),
                'probabilities': {c: round(float(p), 4) for c, p in zip(classes, probs)}
            })
        except Exception as e:
            results.append({'error': str(e)})

    return jsonify({
        'results': results,
        'summary': {
            'total':   len(results),
            'attacks': sum(1 for r in results if r.get('is_attack')),
            'benign':  sum(1 for r in results if not r.get('is_attack') and 'error' not in r),
            'model':   active_model_key
        }
    })


if __name__ == '__main__':
    print('\n🔐 SecureNet IDS API starting...')
    print('   http://127.0.0.1:5000\n')
    app.run(debug=True, port=5000)