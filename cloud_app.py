# ============================================================
#  cloud_app.py — Industrial Mobile Cloud ML System
#  SpamGuard Cloud Server
#
#  Endpoints:
#    GET  /                  — Admin dashboard
#    POST /api/predict       — Single message prediction
#    POST /api/batch         — Multiple messages
#    GET  /api/queue?n=20    — Recent messages feed
#    GET  /api/stats         — System statistics
#    GET  /api/devices       — Device registry
#    GET  /health            — Health check (Render uses this)
# ============================================================

import os, re, sys, json, pickle, time, uuid, threading
from datetime import datetime
from collections import deque
import numpy as np
from flask import Flask, render_template, request, jsonify
from enhance2_combined_model import StackedEnsemble, ConceptDriftDetector  # noqa

# ── Paths (always absolute) ───────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "data", "combined_model.pkl")
CLEAN_PATH = os.path.join(BASE, "data", "cleaned_data.csv")
sys.path.insert(0, BASE)

app = Flask(__name__,
            template_folder=os.path.join(BASE, "templates"),
            static_folder   =os.path.join(BASE, "static"))

# ── Spam detection threshold ──────────────────────────────────
SPAM_THRESHOLD = 0.35   # lower catches subtle/Indian spam

# ── In-memory message queue (thread-safe) ────────────────────
MESSAGE_QUEUE   = deque(maxlen=500)
DEVICE_REGISTRY = {}
LOCK            = threading.Lock()

stats = {
    "total"    : 0,
    "spam"     : 0,
    "ham"      : 0,
    "started"  : time.time(),
}

# ── Model globals ─────────────────────────────────────────────
model_data   = None
fresh_scaler = None
feat_names   = None

# ── Stop words ────────────────────────────────────────────────
STOP = {
    "a","an","the","and","or","but","is","are","was","were","be","been",
    "being","have","has","had","do","does","did","will","would","shall",
    "should","may","might","must","can","could","not","no","i","me","my",
    "we","our","you","your","he","she","it","its","they","them","their",
    "this","that","to","of","in","for","with","on","at","by","from",
    "up","out","s","t","ll",
}

def clean(text):
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    return " ".join(w for w in text.split() if w not in STOP and len(w) > 1)


# ════════════════════════════════════════════════════════════
#  MODEL LOADER
# ════════════════════════════════════════════════════════════
def load_model():
    global model_data, fresh_scaler, feat_names

    if not os.path.exists(MODEL_PATH):
        print(f"\n  ⚠️  Model not found: {MODEL_PATH}")
        print("  Run start_cloud.py first to copy the trained model.\n")
        return False

    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    from enhance1_behavioral_features import extract_behavioral_features
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    sample     = extract_behavioral_features("test")
    feat_names = list(sample.keys())

    if os.path.exists(CLEAN_PATH):
        df   = pd.read_csv(CLEAN_PATH)
        rows = df["clean_message"].fillna("").apply(extract_behavioral_features)
        fd   = pd.DataFrame(list(rows))[feat_names]
        fd   = fd.replace([float("inf"), float("-inf")], 0).fillna(0)
        fresh_scaler = MinMaxScaler()
        fresh_scaler.fit(fd)
    else:
        fresh_scaler = MinMaxScaler()
        fresh_scaler.fit(np.zeros((2, len(feat_names))))

    r = model_data.get("results", {})
    print(f"  ✅ Model loaded  |  Acc: {r.get('Accuracy',0):.1f}%  "
          f"F1: {r.get('F1-Score',0):.1f}%  |  {len(feat_names)} features")
    return True


# ════════════════════════════════════════════════════════════
#  PREDICTION ENGINE
# ════════════════════════════════════════════════════════════
def predict(message: str, device_id: str = "unknown") -> dict:
    if model_data is None:
        return {"error": "Model not loaded"}

    from enhance1_behavioral_features import extract_behavioral_features

    ensemble   = model_data["ensemble"]
    vectorizer = model_data["vectorizer"]

    X_tfidf  = vectorizer.transform([clean(message)])
    bfeats   = extract_behavioral_features(message)
    X_b_raw  = np.array([[bfeats[f] for f in feat_names]], dtype=float)
    X_b_raw  = np.nan_to_num(X_b_raw)
    X_b      = fresh_scaler.transform(X_b_raw)

    proba    = ensemble.predict_proba(X_tfidf, X_b)[0]
    is_spam  = bool(proba[1] >= SPAM_THRESHOLD)
    spam_pct = round(float(proba[1]) * 100, 1)

    # Top words
    feat_out  = vectorizer.get_feature_names_out()
    tfidf_arr = X_tfidf.toarray()[0]
    top_words = [feat_out[i] for i in np.argsort(tfidf_arr)[::-1]
                 if tfidf_arr[i] > 0][:5]

    result = {
        "id"        : str(uuid.uuid4())[:8],
        "timestamp" : datetime.now().strftime("%H:%M:%S"),
        "device_id" : device_id,
        "message"   : message,
        "prediction": "SPAM" if is_spam else "LEGITIMATE",
        "spam_prob" : spam_pct,
        "ham_prob"  : round(float(proba[0]) * 100, 1),
        "top_words" : top_words,
        "is_spam"   : is_spam,
    }

    # Update queue + stats + device registry
    with LOCK:
        MESSAGE_QUEUE.appendleft(result)
        stats["total"] += 1
        if is_spam:
            stats["spam"] += 1
        else:
            stats["ham"] += 1
        stats.setdefault("devices", set()).add(device_id)

        prev = DEVICE_REGISTRY.get(device_id, {"total": 0, "spam": 0})
        spam_count = prev["spam"] + (1 if is_spam else 0)
        DEVICE_REGISTRY[device_id] = {
            "last_seen": result["timestamp"],
            "total"    : prev["total"] + 1,
            "spam"     : spam_count,
            "status"   : "BLOCKED" if spam_count >= 3 else "ACTIVE",
        }

    # Security alert for blocked devices
    if DEVICE_REGISTRY[device_id]["status"] == "BLOCKED":
        print(f"  🚨 SECURITY ALERT: {device_id} BLOCKED at {result['timestamp']}")

    return result


# ════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════
@app.route("/")
def dashboard():
    uptime  = int(time.time() - stats["started"])
    h, m    = divmod(uptime // 60, 60)
    up_str  = f"{h}h {m}m" if h else f"{m}m"
    model_ok = model_data is not None
    perf    = model_data["results"] if model_ok else {}
    return render_template("dashboard.html",
                           stats=stats, uptime=up_str,
                           model_loaded=model_ok, perf=perf)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data    = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Provide 'message' field"}), 400
    device  = data.get("device_id", f"device-{request.remote_addr}")
    return jsonify(predict(message, device))


@app.route("/api/batch", methods=["POST"])
def api_batch():
    data     = request.get_json(silent=True) or {}
    messages = data.get("messages", [])
    device   = data.get("device_id", "batch-client")
    results  = [predict(m, device) for m in messages if m.strip()]
    return jsonify({
        "total"     : len(results),
        "spam"      : sum(1 for r in results if r["is_spam"]),
        "legitimate": sum(1 for r in results if not r["is_spam"]),
        "results"   : results,
    })


@app.route("/api/queue")
def api_queue():
    n = min(int(request.args.get("n", 30)), 100)
    with LOCK:
        msgs = list(MESSAGE_QUEUE)[:n]
    return jsonify({"messages": msgs, "total": stats["total"]})


@app.route("/api/stats")
def api_stats():
    uptime   = int(time.time() - stats["started"])
    total    = stats["total"]
    spam_pct = round(stats["spam"] / max(total, 1) * 100, 1)
    return jsonify({
        "total"         : total,
        "spam"          : stats["spam"],
        "ham"           : stats["ham"],
        "spam_rate"     : spam_pct,
        "devices_active": len(DEVICE_REGISTRY),
        "uptime_seconds": uptime,
        "model_loaded"  : model_data is not None,
    })


@app.route("/api/devices")
def api_devices():
    with LOCK:
        devices = dict(DEVICE_REGISTRY)
    return jsonify({"devices": devices})


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model" : "loaded" if model_data else "not loaded",
        "uptime": int(time.time() - stats["started"]),
    })


# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  🌐 SpamGuard — Industrial Mobile Cloud Server")
    print("=" * 55)
    load_model()
    port = int(os.environ.get("PORT", 5001))
    print(f"\n  Dashboard : http://localhost:{port}")
    print(f"  API       : http://localhost:{port}/api/predict")
    print(f"  Health    : http://localhost:{port}/health")
    print(f"\n  Open a NEW terminal and run:")
    print(f"  python device_simulator.py")
    print(f"\n  Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=port, debug=False)