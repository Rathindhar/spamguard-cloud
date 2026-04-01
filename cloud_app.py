# ============================================================
#  cloud_app.py — SpamGuard Cloud (Render version)
#  Uses self-contained model trained by train_on_render.py
# ============================================================

import os, re, sys, pickle, time, uuid, threading
from datetime import datetime
from collections import deque
import numpy as np
from flask import Flask, render_template, request, jsonify
from scipy.sparse import hstack, csr_matrix

BASE       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "data", "combined_model.pkl")
sys.path.insert(0, BASE)

app = Flask(__name__,
            template_folder=os.path.join(BASE, "templates"),
            static_folder   =os.path.join(BASE, "static"))

SPAM_THRESHOLD  = 0.35
MESSAGE_QUEUE   = deque(maxlen=500)
DEVICE_REGISTRY = {}
LOCK            = threading.Lock()
stats = {"total":0,"spam":0,"ham":0,"started":time.time()}

model_data = None

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

def get_behavioral(text, spam_kw, ham_kw):
    if not text: return np.zeros((1,5))
    words     = [w.strip("!.,?:()").lower() for w in text.split()]
    spam_hits = sum(1 for w in words if w in spam_kw)
    ham_hits  = sum(1 for w in words if w in ham_kw)
    url_count = len(re.findall(r"http|www\.|\.com", text, re.I))
    caps      = sum(1 for c in text if c.isupper()) / max(len(text),1)
    exclaim   = text.count("!")
    return np.array([[max(0, spam_hits-ham_hits), ham_hits,
                      url_count, caps, exclaim]], dtype=float)

def load_model():
    global model_data
    print(f"\n  Model path: {MODEL_PATH}")
    print(f"  Exists: {os.path.exists(MODEL_PATH)}")
    if os.path.exists(BASE+'/data'):
        print(f"  data/ contents: {os.listdir(BASE+'/data')}")

    if not os.path.exists(MODEL_PATH):
        print("  WARNING: Model not found — predictions unavailable")
        return False
    try:
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
        r = model_data.get("results", {})
        print(f"  Model loaded! Acc={r.get('Accuracy',0):.1f}% F1={r.get('F1-Score',0):.1f}%")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        return False

def predict(message, device_id="unknown"):
    if model_data is None:
        return {"message":message,"prediction":"MODEL NOT LOADED",
                "spam_prob":0,"ham_prob":0,"top_words":[],
                "is_spam":False,"timestamp":datetime.now().strftime("%H:%M:%S"),
                "device_id":device_id,"id":str(uuid.uuid4())[:8]}

    best_name  = model_data["best_name"]
    clf        = model_data["classifiers"][best_name]
    vectorizer = model_data["vectorizer"]
    scaler     = model_data["behavioral_scaler"]
    spam_kw    = model_data["spam_keywords"]
    ham_kw     = model_data["ham_keywords"]

    X_tfidf = vectorizer.transform([clean(message)])
    X_b_raw = get_behavioral(message, spam_kw, ham_kw)
    X_b_raw = np.nan_to_num(X_b_raw)
    X_b     = scaler.transform(X_b_raw)

    if "Naive" in best_name:
        proba = clf.predict_proba(X_tfidf)[0]
    else:
        X_c   = hstack([X_tfidf, csr_matrix(X_b)])
        proba = clf.predict_proba(X_c)[0]

    is_spam  = bool(proba[1] >= SPAM_THRESHOLD)
    spam_pct = round(float(proba[1]) * 100, 1)

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

    with LOCK:
        MESSAGE_QUEUE.appendleft(result)
        stats["total"] += 1
        stats["spam" if is_spam else "ham"] += 1
        prev   = DEVICE_REGISTRY.get(device_id, {"total":0,"spam":0})
        spam_c = prev["spam"] + (1 if is_spam else 0)
        DEVICE_REGISTRY[device_id] = {
            "last_seen": result["timestamp"],
            "total"    : prev["total"] + 1,
            "spam"     : spam_c,
            "status"   : "BLOCKED" if spam_c >= 3 else "ACTIVE",
        }
    if DEVICE_REGISTRY[device_id]["status"] == "BLOCKED":
        print(f"  ALERT: {device_id} BLOCKED at {result['timestamp']}")
    return result

@app.route("/")
def dashboard():
    uptime = int(time.time() - stats["started"])
    h, m   = divmod(uptime // 60, 60)
    perf   = model_data.get("results", {}) if model_data else {}
    return render_template("dashboard.html", stats=stats,
                           uptime=f"{h}h {m}m" if h else f"{m}m",
                           model_loaded=model_data is not None, perf=perf)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data    = request.get_json(silent=True) or {}
    message = data.get("message","").strip()
    if not message: return jsonify({"error":"Provide message field"}), 400
    return jsonify(predict(message, data.get("device_id", request.remote_addr)))

@app.route("/api/batch", methods=["POST"])
def api_batch():
    data    = request.get_json(silent=True) or {}
    msgs    = data.get("messages", [])
    device  = data.get("device_id", "batch-client")
    results = [predict(m, device) for m in msgs if m.strip()]
    return jsonify({"total":len(results),
                    "spam":sum(1 for r in results if r["is_spam"]),
                    "legitimate":sum(1 for r in results if not r["is_spam"]),
                    "results":results})

@app.route("/api/queue")
def api_queue():
    n = min(int(request.args.get("n",30)), 100)
    with LOCK: msgs = list(MESSAGE_QUEUE)[:n]
    return jsonify({"messages":msgs,"total":stats["total"]})

@app.route("/api/stats")
def api_stats():
    total = stats["total"]
    return jsonify({"total":total,"spam":stats["spam"],"ham":stats["ham"],
                    "spam_rate":round(stats["spam"]/max(total,1)*100,1),
                    "devices_active":len(DEVICE_REGISTRY),
                    "uptime_seconds":int(time.time()-stats["started"]),
                    "model_loaded":model_data is not None})

@app.route("/api/devices")
def api_devices():
    with LOCK: devices = dict(DEVICE_REGISTRY)
    return jsonify({"devices":devices})

@app.route("/health")
def health():
    return jsonify({"status":"healthy",
                    "model":"loaded" if model_data else "not loaded",
                    "uptime":int(time.time()-stats["started"])})

load_model()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"\n  Dashboard: http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
