# ============================================================
#  device_simulator.py  —  Mobile Device Simulator
#
#  Simulates 5 virtual mobile devices in an industrial
#  environment continuously sending messages to your cloud
#  ML server via REST API.
#
#  Each device has a personality:
#   Device-001  Regular user      mostly legitimate messages
#   Device-002  Spammer           mostly spam messages
#   Device-003  Mixed user        50/50 spam and legitimate
#   Device-004  IoT sensor        automated status messages
#   Device-005  Compromised       starts clean, goes spammy
#
#  Usage:
#    Local:  python device_simulator.py
#    Cloud:  python device_simulator.py --url https://your-app.onrender.com
# ============================================================

import time
import random
import argparse
import threading
import requests
from datetime import datetime

# ── Parse arguments ───────────────────────────────────────────
parser = argparse.ArgumentParser(description="Mobile Device Simulator")
parser.add_argument("--url", default="http://localhost:5001",
                    help="Cloud server URL (default: http://localhost:5001)")
parser.add_argument("--rate", type=float, default=2.0,
                    help="Messages per second per device (default: 2.0)")
parser.add_argument("--devices", type=int, default=5,
                    help="Number of simulated devices (default: 5)")
args = parser.parse_args()

API_URL    = f"{args.url.rstrip('/')}/api/predict"
MSG_RATE   = args.rate
N_DEVICES  = args.devices

# ── Message pools ─────────────────────────────────────────────
LEGITIMATE_MESSAGES = [
    "Hey, are we still meeting for lunch today?",
    "Can you send me the notes from yesterday's class?",
    "I will reach the office by 10 AM tomorrow.",
    "Mom asked if you can buy milk on the way home.",
    "Happy birthday! Hope you have a great day.",
    "Let's study together for the exam this weekend.",
    "The meeting has been moved to 3 PM.",
    "I sent you the project files via email.",
    "Please call me when you reach home.",
    "Don't forget to bring the charger tomorrow.",
    "What time does the movie start?",
    "I will send the assignment tonight.",
    "Thanks for helping me with the coding problem.",
    "Let's go for cricket practice in the evening.",
    "The professor uploaded the lecture slides.",
    "I just reached the library.",
    "Can you help me debug this Java code?",
    "Let's have dinner together tonight.",
    "The exam results will be announced tomorrow.",
    "Please remind me to submit the assignment tonight.",
    "Are you coming to the college event today?",
    "I will share the GitHub repository link soon.",
    "Let's prepare together for the coding interview.",
    "I finished solving that recursion problem.",
    "Good luck for your interview tomorrow!",
    "System status: All sensors operational. Temperature normal.",
    "Device check-in: Battery 85%, Signal strength good.",
    "Alert: Door sensor triggered at Zone-3.",
    "Maintenance scheduled for Unit-7 at 14:00.",
]

SPAM_MESSAGES = [
    "Congratulations! You have won ₹10,000 cash. Claim now at http://win-prize.com",
    "URGENT! Your bank account will be blocked. Verify immediately: http://bank-alert.net",
    "Get an iPhone 15 for just ₹999 today! Limited offer. Click now.",
    "Earn ₹5000 daily from home with this simple trick. Register now.",
    "Claim your free Amazon gift card worth ₹5000 here: http://gift-now.com",
    "Your Paytm KYC is pending. Update now to avoid suspension.",
    "Limited time loan offer with zero interest. Apply today.",
    "Free Netflix subscription for 3 months! Activate now.",
    "Congratulations! You are selected for a government subsidy scheme.",
    "Win a brand new car by participating in our lucky draw today.",
    "Click here to get unlimited free mobile data instantly.",
    "Hot investment opportunity with guaranteed profits. Join now.",
    "Your parcel delivery failed. Update address: http://parcel-update.com",
    "Free recharge worth ₹500 available today only. Claim fast.",
    "Your SIM card will be deactivated in 24 hours. Update details now.",
    "Exclusive crypto investment opportunity with 200% returns.",
    "You have won a free vacation to Dubai! Claim now.",
    "Your Google account has suspicious activity. Verify now.",
    "Click here to get 90% discount on branded clothes.",
    "Earn money quickly with this secret online method.",
    "Free laptop under government student scheme. Apply now.",
    "Activate your premium membership for free today only.",
    "Your email storage is full. Upgrade immediately.",
    "Download this app and earn money daily.",
    "Last chance to claim your ₹20,000 cashback reward.",
]

# ── Device personalities ──────────────────────────────────────
DEVICE_PROFILES = [
    {"id": "Device-001", "name": "Regular User",   "spam_ratio": 0.05, "color": "\033[92m"},
    {"id": "Device-002", "name": "Spammer",        "spam_ratio": 0.90, "color": "\033[91m"},
    {"id": "Device-003", "name": "Mixed User",     "spam_ratio": 0.50, "color": "\033[93m"},
    {"id": "Device-004", "name": "IoT Sensor",     "spam_ratio": 0.02, "color": "\033[96m"},
    {"id": "Device-005", "name": "Compromised",    "spam_ratio": 0.70, "color": "\033[95m"},
]

RESET  = "\033[0m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"

sent_count  = 0
spam_count  = 0
error_count = 0
lock        = threading.Lock()


def send_message(device: dict):
    """Send one message from a device to the cloud server"""
    global sent_count, spam_count, error_count

    # Choose message based on device spam ratio
    if random.random() < device["spam_ratio"]:
        message = random.choice(SPAM_MESSAGES)
    else:
        message = random.choice(LEGITIMATE_MESSAGES)

    try:
        resp = requests.post(
            API_URL,
            json={"message": message, "device_id": device["id"]},
            timeout=10
        )
        if resp.status_code == 200:
            result = resp.json()
            pred   = result.get("prediction", "UNKNOWN")
            prob   = result.get("spam_prob", 0)
            ts     = datetime.now().strftime("%H:%M:%S")

            with lock:
                sent_count += 1
                if pred == "SPAM":
                    spam_count += 1

            # Console output
            icon  = "🚨" if pred == "SPAM" else "✅"
            color = RED if pred == "SPAM" else GREEN
            print(f"  [{ts}] {device['color']}{device['id']}{RESET} "
                  f"({device['name']:<12}) "
                  f"{color}{icon} {pred:<12}{RESET} "
                  f"({prob:>5.1f}%)  "
                  f"\"{message[:45]}{'...' if len(message)>45 else ''}\"")
        else:
            with lock:
                error_count += 1
            print(f"  {RED}[ERROR]{RESET} {device['id']} got HTTP {resp.status_code}")

    except requests.exceptions.ConnectionError:
        with lock:
            error_count += 1
        if error_count == 1:
            print(f"\n  {RED}[ERROR]{RESET} Cannot connect to {API_URL}")
            print(f"  Make sure the cloud server is running first.\n")
    except Exception as e:
        with lock:
            error_count += 1


def device_loop(device: dict, interval: float):
    """Continuously send messages at the given interval"""
    while True:
        send_message(device)
        # Add slight randomness to message timing
        time.sleep(interval * random.uniform(0.7, 1.3))


def print_summary():
    """Print stats every 30 seconds"""
    while True:
        time.sleep(30)
        with lock:
            total = sent_count
            spam  = spam_count
        if total > 0:
            rate = spam / total * 100
            print(f"\n  {'─'*60}")
            print(f"  📊 {BOLD}SUMMARY{RESET}  |  "
                  f"Total: {total}  |  "
                  f"Spam: {spam} ({rate:.1f}%)  |  "
                  f"Legit: {total-spam}")
            print(f"  {'─'*60}\n")


def main():
    print(f"\n{'='*65}")
    print(f"  🌐 {BOLD}Industrial Mobile Cloud — Device Simulator{RESET}")
    print(f"{'='*65}")
    print(f"  Cloud server : {args.url}")
    print(f"  API endpoint : {API_URL}")
    print(f"  Devices      : {N_DEVICES}")
    print(f"  Message rate : {MSG_RATE}/sec per device")
    print(f"\n  Starting {N_DEVICES} virtual mobile devices...")
    print(f"  Press Ctrl+C to stop.\n")
    print(f"  {'─'*63}")
    print(f"  {'Time':<10} {'Device':<12} {'Type':<14} "
          f"{'Result':<14} {'Prob':>6}  {'Message'}")
    print(f"  {'─'*63}\n")

    interval = 1.0 / MSG_RATE
    threads  = []

    for i, profile in enumerate(DEVICE_PROFILES[:N_DEVICES]):
        t = threading.Thread(
            target=device_loop,
            args=(profile, interval),
            daemon=True
        )
        t.start()
        threads.append(t)
        time.sleep(0.3)  # stagger startup

    # Summary thread
    s = threading.Thread(target=print_summary, daemon=True)
    s.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n\n  Simulation stopped.")
        print(f"  Final: {sent_count} messages sent, "
              f"{spam_count} spam detected ({spam_count/max(sent_count,1)*100:.1f}%)")


if __name__ == "__main__":
    main()