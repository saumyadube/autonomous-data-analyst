"""
test_api.py
============
Run this script to check if your Groq API key is working correctly.

Usage:
    python test_api.py

If it prints "API KEY WORKS" — your app will work fine.
If it shows an error — follow the fix instructions printed.
"""

import os
import sys

print("=" * 55)
print("  Groq API Key Tester")
print("  Saumaya Dube | Rama University Capstone Project")
print("=" * 55)

# Step 1: Load .env
from dotenv import load_dotenv
load_dotenv(override=True)

api_key = os.environ.get("GROQ_API_KEY", "").strip()

print(f"\n[1] Checking .env file...")
if not api_key or api_key == "your_groq_api_key_here":
    print("    ❌ API key NOT found in .env file")
    print("\n    FIX: Open the .env file and change:")
    print("         GROQ_API_KEY=your_groq_api_key_here")
    print("    to your real key:")
    print("         GROQ_API_KEY=gsk_abc123yourRealKeyHere")
    print("\n    Get a FREE key at: https://console.groq.com")
    sys.exit(1)
else:
    # Mask the key for safety
    masked = api_key[:8] + "..." + api_key[-4:]
    print(f"    ✅ API key found: {masked}")

# Step 2: Check groq package
print(f"\n[2] Checking groq package...")
try:
    from groq import Groq
    print("    ✅ groq package installed")
except ImportError:
    print("    ❌ groq package NOT installed")
    print("\n    FIX: Run this command:")
    print("         pip install groq")
    sys.exit(1)

# Step 3: Test actual API call
print(f"\n[3] Testing API connection (sending test message)...")
try:
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Say: API_TEST_OK"}],
        max_tokens=20,
        temperature=0,
    )
    reply = response.choices[0].message.content.strip()
    print(f"    ✅ API responded: {reply}")
except Exception as e:
    print(f"    ❌ API call failed: {e}")
    print("\n    Common fixes:")
    print("    - Check your internet connection")
    print("    - Make sure the API key is correct (no extra spaces)")
    print("    - Try generating a new key at https://console.groq.com")
    sys.exit(1)

print("\n" + "=" * 55)
print("  ✅  API KEY WORKS — Your app is ready to run!")
print("  Run:  streamlit run app.py")
print("=" * 55 + "\n")
