# ============================================================
#  start_cloud.py — Local Cloud Launcher
#
#  Run this ONCE to set up, then use it every time to start.
#  It copies the trained model from spammer_enhanced/ and
#  launches the cloud server at http://localhost:5001
#
#  In a SECOND terminal, run:  python device_simulator.py
# ============================================================

import os, sys, shutil, subprocess

BASE         = os.path.dirname(os.path.abspath(__file__))
ENHANCED_DIR = os.path.join(os.path.dirname(BASE), "spammer_enhanced")
DATA_DIR     = os.path.join(BASE, "data")

def banner(t):
    print(f"\n{'='*55}\n  {t}\n{'='*55}")

def copy_model_files():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Files to copy from spammer_enhanced
    files = [
        ("data/combined_model.pkl",            "combined_model.pkl"),
        ("data/cleaned_data.csv",              "cleaned_data.csv"),
        ("enhance1_behavioral_features.py",    "enhance1_behavioral_features.py"),
        ("enhance2_combined_model.py",         "enhance2_combined_model.py"),
    ]

    all_ok = True
    for src_rel, dst_name in files:
        src = os.path.join(ENHANCED_DIR, src_rel)
        # Determine destination
        if dst_name.endswith(".pkl") or dst_name.endswith(".csv"):
            dst = os.path.join(DATA_DIR, dst_name)
        else:
            dst = os.path.join(BASE, dst_name)

        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✅ Copied: {dst_name}")
        else:
            print(f"  ❌ Missing: {src_rel}")
            if ".pkl" in dst_name:
                all_ok = False

    return all_ok

def check_requirements():
    try:
        import flask, sklearn, numpy, pandas
        print("  ✅ All packages available")
        return True
    except ImportError as e:
        print(f"  ❌ Missing package: {e}")
        print("  Run: pip install flask scikit-learn pandas numpy scipy")
        return False

if __name__ == "__main__":
    banner("SpamGuard Cloud — Setup & Launch")

    print("\n[1/3] Checking packages...")
    if not check_requirements():
        sys.exit(1)

    print("\n[2/3] Copying model files from spammer_enhanced/...")
    if not copy_model_files():
        print("\n  ⚠️  Model file missing!")
        print("  Run spammer_enhanced/run_all.py first to train the model.")
        sys.exit(1)

    print("\n[3/3] Starting cloud server...")
    banner("SpamGuard Cloud is Running!")
    print("\n  📊 Dashboard : http://localhost:5001")
    print("  📡 API       : http://localhost:5001/api/predict")
    print("  ❤️  Health    : http://localhost:5001/health")
    print("\n  ─────────────────────────────────────────")
    print("  Open a NEW terminal window and run:")
    print("  python device_simulator.py")
    print("  ─────────────────────────────────────────")
    print("\n  Press Ctrl+C to stop the server\n")

    app_path = os.path.join(BASE, "cloud_app.py")
    os.chdir(BASE)
    os.execv(sys.executable, [sys.executable, app_path])