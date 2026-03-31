# SpamGuard Cloud — Industrial Mobile Cloud System
### Full Deployment Guide: Local → Render.com (Free)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│          INDUSTRIAL MOBILE CLOUD SYSTEM                 │
│                                                         │
│  📱 Device Simulator        🌐 Cloud Server (Render)    │
│  ┌──────────────────┐       ┌────────────────────────┐  │
│  │ Device-001 (User)│──────▶│  REST API /api/predict │  │
│  │ Device-002 (Spam)│──────▶│  Message Queue         │  │
│  │ Device-003 (Mix) │──────▶│  ML Engine (Ensemble)  │  │
│  │ Device-004 (IoT) │──────▶│  Device Registry       │  │
│  │ Device-005 (Comp)│──────▶│  Real-time Dashboard   │  │
│  └──────────────────┘       └────────────────────────┘  │
│                                      │                  │
│                              ┌───────▼───────┐          │
│                              │ Admin Browser │          │
│                              │ (live alerts) │          │
│                              └───────────────┘          │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Files in this folder

```
spammer_cloud/
├── cloud_app.py            ← Main Flask cloud server
├── device_simulator.py     ← Simulates 5 mobile devices
├── start_cloud.py          ← Local launcher (copies model files)
├── requirements.txt        ← Python packages for Render
├── render.yaml             ← Render deployment config
├── templates/
│   └── dashboard.html      ← Real-time admin dashboard
└── README.md               ← This file
```

---

## 🖥️ PHASE 1 — Run Locally First

### Step 1: Start cloud server
```powershell
cd C:\Users\suchi\Documents\SDITS\spammer_cloud
python start_cloud.py
```
Open browser: **http://localhost:5000**

### Step 2: Start device simulator (new terminal)
```powershell
cd C:\Users\suchi\Documents\SDITS\spammer_cloud
python device_simulator.py
```
Watch 5 virtual devices send spam and legitimate messages live!

### Step 3: Test the REST API manually
```powershell
curl -X POST http://localhost:5000/api/predict `
  -H "Content-Type: application/json" `
  -d "{\"message\": \"Congratulations! You won Rs.10000!\", \"device_id\": \"my-phone\"}"
```

---

## ☁️ PHASE 2 — Deploy to Render.com (Free, Live URL)

### Step 1: Install Git (if not installed)
Download from: https://git-scm.com/download/win

### Step 2: Create GitHub repository
1. Go to https://github.com and sign in (or create free account)
2. Click **New repository**
3. Name it: `spamguard-cloud`
4. Make it **Public**
5. Click **Create repository**

### Step 3: Upload your project to GitHub
Open PowerShell in your `SDITS` folder:
```powershell
cd C:\Users\suchi\Documents\SDITS

# Copy cloud files + required model files
mkdir spamguard-cloud
xcopy spammer_cloud\* spamguard-cloud\ /E /I
xcopy spammer_enhanced\enhance1_behavioral_features.py spamguard-cloud\
xcopy spammer_enhanced\enhance2_combined_model.py spamguard-cloud\
mkdir spamguard-cloud\data
xcopy spammer_enhanced\data\combined_model.pkl spamguard-cloud\data\
xcopy spammer_enhanced\data\cleaned_data.csv spamguard-cloud\data\

# Initialize git and push
cd spamguard-cloud
git init
git add .
git commit -m "SpamGuard Cloud - Industrial ML Spammer Detection"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/spamguard-cloud.git
git push -u origin main
```
> Replace YOUR_USERNAME with your GitHub username

### Step 4: Deploy on Render
1. Go to **https://render.com** and sign up free (use Google)
2. Click **New** → **Web Service**
3. Click **Connect a repository** → select `spamguard-cloud`
4. Render auto-detects `render.yaml` and fills in settings
5. Click **Create Web Service**
6. Wait 3-5 minutes for deployment ✅

### Step 5: Get your live URL
Render gives you a URL like:
```
https://spamguard-cloud.onrender.com
```
Share this with your professor — it works on any phone or browser!

### Step 6: Point device simulator at cloud URL
```powershell
python device_simulator.py --url https://spamguard-cloud.onrender.com
```
Now your virtual devices are sending messages to the real internet cloud!

---

## 🌐 API Reference

### Predict single message
```
POST /api/predict
{"message": "text here", "device_id": "device-001"}

Response:
{"prediction": "SPAM", "spam_prob": 98.5, "ham_prob": 1.5, ...}
```

### Batch prediction
```
POST /api/batch
{"messages": ["msg1", "msg2", ...], "device_id": "batch-client"}
```

### System stats
```
GET /api/stats
{"total_processed": 142, "spam_detected": 67, "spam_rate": 47.2, ...}
```

### Device registry
```
GET /api/devices
{"devices": {"device-001": {"status": "ACTIVE", "spam_sent": 2, ...}}}
```

### Health check
```
GET /health
{"status": "healthy", "model": "loaded", "uptime": 3600}
```

---

## 🎓 What to say in your viva

> "Our system implements a three-layer industrial mobile cloud architecture.
> Layer one consists of mobile edge devices — represented here by our device
> simulator running 5 virtual endpoints with different behavioral profiles.
> 
> Layer two is our cloud ML engine deployed on Render, which exposes a REST API
> that any device can call. It maintains an in-memory message queue, processes
> each message through our stacked ensemble model, and updates the device
> registry in real time.
>
> Layer three is this admin dashboard — security personnel can monitor live
> threat detection, see which devices are flagged as compromised, and test
> any message manually through the same cloud API the devices use.
>
> This matches exactly the architecture described in Section III of our paper."

---

## ⚠️ Render Free Tier Notes

- Free tier sleeps after 15 minutes of inactivity
- First request after sleep takes ~30 seconds to wake up
- To keep it awake during demo: visit the URL every few minutes
- Upgrade to Starter ($7/month) for always-on if needed
