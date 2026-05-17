# 📊 LLM-Powered Autonomous Data Analyst

**Student:** Saumaya Dube  
**College:** Rama University, Kanpur, Uttar Pradesh  
**Project:** Capstone Project 2025–2026

---

## 🎯 What This Project Does

You upload any CSV file → The AI automatically:
1. Reads and understands your data
2. Runs 8 statistical analysis tools
3. Generates charts and visualizations
4. Writes a complete business report
5. Lets you download a PDF report

**No coding needed to USE it — just upload and click!**

---

## 🖥️ STEP-BY-STEP SETUP ON WINDOWS

### ✅ STEP 1: Install Python

1. Go to **https://www.python.org/downloads/**
2. Download **Python 3.11** (click the big yellow button)
3. Run the installer
4. ⚠️ **IMPORTANT:** Check the box **"Add Python to PATH"** before clicking Install
5. Click **Install Now**
6. Wait for it to finish

**To verify Python is installed:**
- Press `Windows + R`, type `cmd`, press Enter
- Type: `python --version`
- You should see: `Python 3.11.x`

---

### ✅ STEP 2: Get Your FREE Groq API Key

This project uses **Groq** (100% FREE, no credit card):

1. Go to **https://console.groq.com**
2. Click **"Sign Up"** (use Google or GitHub login)
3. After login, click **"API Keys"** in the left menu
4. Click **"Create API Key"**
5. Give it a name (e.g., "DataAnalyst")
6. **Copy the key** (it starts with `gsk_...`)
7. **Save it somewhere** — you'll need it in Step 4

---

### ✅ STEP 3: Extract the Project

1. Find the downloaded ZIP file: `autonomous_data_analyst.zip`
2. Right-click → **Extract All**
3. Choose a location (e.g., `C:\Users\YourName\Desktop\`)
4. Open the extracted folder: `autonomous_data_analyst`

---

### ✅ STEP 4: Add Your API Key

1. Open the file called **`.env`** (use Notepad)
   - If you can't see it, in File Explorer: View → Show → Hidden Items
2. Find this line:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
3. Replace `your_groq_api_key_here` with your actual key:
   ```
   GROQ_API_KEY=gsk_abc123yourActualKeyHere
   ```
4. Save the file (Ctrl+S)

---

### ✅ STEP 5: Install Required Packages

1. Open the `autonomous_data_analyst` folder
2. Double-click **`setup.bat`**
3. A black window (Command Prompt) will open
4. Wait for it to finish (takes 3–5 minutes first time)
5. When you see "Setup complete!" — you're done!

**What this does:** Creates a virtual environment and installs all Python libraries automatically.

---

### ✅ STEP 6: Run the App

**Option A (Easy):** Double-click **`run.bat`**

**Option B (Manual):**
1. Open Command Prompt in the project folder:
   - Hold `Shift` + Right-click in the folder → "Open PowerShell window here"
2. Type these commands one by one:
   ```
   venv\Scripts\activate
   streamlit run app.py
   ```

**After a few seconds, your browser will automatically open to:**
```
http://localhost:8501
```

---

### ✅ STEP 7: Use the App

1. **Enter your Groq API key** in the sidebar (or add it to .env)
2. **Upload a CSV file** (or click "Load Sample Dataset")
3. **(Optional)** Type your analysis goal
4. Click **"🚀 Run Analysis"**
5. Wait 1–3 minutes for the analysis
6. View your **charts, AI report, and download PDF**

---

## 📁 Project File Structure

```
autonomous_data_analyst/
│
├── app.py                    ← Main app (run this!)
├── requirements.txt          ← All Python packages needed
├── .env                      ← Your API key goes here
├── setup.bat                 ← Windows setup script
├── run.bat                   ← Windows run script
│
├── src/
│   ├── ingestion/
│   │   └── profiler.py       ← Reads CSV, creates dataset profile
│   │
│   ├── tools/
│   │   └── analytics.py      ← 8 statistical analysis tools
│   │
│   ├── synthesis/
│   │   └── agent.py          ← LLM (AI) planning and report generation
│   │
│   └── report/
│       └── generator.py      ← PDF report creation
│
└── data/sample/              ← Put sample CSV files here
```

---

## 🔧 Troubleshooting

### ❌ "Python not found"
→ Reinstall Python and make sure you check "Add Python to PATH"

### ❌ "pip is not recognized"
→ Open Command Prompt as Administrator and run:
```
python -m pip install --upgrade pip
```

### ❌ "ModuleNotFoundError"
→ Make sure you ran `setup.bat` first, then `run.bat`

### ❌ "Groq API key not found"
→ Open `.env` file and make sure your key is there without spaces

### ❌ App opens but shows error on analysis
→ Check your internet connection (Groq API needs internet)

### ❌ Port already in use
→ Run this command: `streamlit run app.py --server.port 8502`

---

## 📦 Libraries Used (All Free & Open Source)

| Library | Purpose |
|---------|---------|
| `streamlit` | Web app interface |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `scipy` | Statistical tests |
| `scikit-learn` | Machine learning tools |
| `matplotlib` | Chart generation |
| `seaborn` | Statistical visualizations |
| `plotly` | Interactive charts |
| `groq` | Free LLM API (LLaMA 3) |
| `fpdf2` | PDF report generation |

---

## 🎓 Academic Information

- **Project Title:** LLM-Powered Autonomous Data Analyst
- **Student:** Saumaya Dube
- **University:** Rama University, Kanpur, UP-209217
- **Department:** Computer Science & Engineering
- **Guide:** Prof. (Dr.) C. S. Raghuvanshi
- **Year:** 2025–2026

---

## 📞 Need Help?

If you get stuck at any step, the most common fixes are:
1. Reinstall Python with "Add to PATH" checked
2. Run `setup.bat` again as Administrator
3. Check your Groq API key in `.env` file
