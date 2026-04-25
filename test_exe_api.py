import subprocess
import time
import requests

p = subprocess.Popen(["dist/app_launcher.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(5)

try:
    with open('samples/download (5).jpg', 'rb') as f:
        resp = requests.post('http://127.0.0.1:8000/api/ocr', files={'file': f}, data={'lang': 'eng'})
    print("EXE API STATUS:", resp.status_code)
    print("EXE API RESPONSE:", resp.text[:500])
except Exception as e:
    print("Error:", e)
finally:
    p.terminate()
