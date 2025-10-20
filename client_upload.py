import requests
from pathlib import Path

API_URL = "http://127.0.0.1:8000/documents"
DOCS_DIR = Path("docs")

if not DOCS_DIR.exists() or not any(DOCS_DIR.iterdir()):
    print(f"Folder {DOCS_DIR} does not exist or is empty.")
    exit(1)

files = [("files", (f.name, open(f, "rb"), "text/plain")) for f in DOCS_DIR.glob("*.*")]

try:
    response = requests.post(API_URL, files=files)
    if response.status_code == 200:
        print("Documents uploaded successfully!")
        print(response.json())
    else:
        print(f"Error {response.status_code}: {response.text}")
finally:
    for _, (_, f_obj, *_ ) in files:
        f_obj.close()
