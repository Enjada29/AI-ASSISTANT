import os
import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

def query_api(question: str):
    """Send a question to the API and display results"""
    data = {"q": question}
    try:
        res = requests.post(f"{API_URL}/query", data=data)
        if res.status_code == 200:
            output = res.json()
            print(f"\nQuery: {output.get('query', '')}")
            print(f"Answer: {output.get('answer', '')}\n")
            print("Top chunks / results:")
            for i, r in enumerate(output.get("results", []), 1):
                if 'chunk' in r:  
                    chunk_preview = r.get("chunk", "")[:200].replace("\n", " ")
                    print(f"{i}. Score: {r.get('score', 0.0):.4f} | {chunk_preview}...")
                else:  
                    print(f"{i}. {r.get('title', '')}")
                    print(f"   {r.get('snippet', '')}")
                    print(f"   URL: {r.get('url', '')}")
            print(f"\nAgent Decision: {output.get('agent_decision', 'N/A')}")
            print(f"Retrieval Confidence: {output.get('retrieval_confidence', 'N/A')}\n")
        else:
            print(f"Error {res.status_code}: {res.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")

def upload_documents(folder_path: str):
    """Upload all files in a folder to the API"""
    folder = Path(folder_path)
    if not folder.exists() or not any(folder.iterdir()):
        print(f"Folder {folder_path} not found or empty.")
        return

    files = [("files", open(f, "rb")) for f in folder.glob("*.*")]
    try:
        res = requests.post(f"{API_URL}/documents", files=files)
        if res.status_code == 200:
            print("Documents uploaded successfully!")
            print(res.json())
        else:
            print(f"Error {res.status_code}: {res.text}")
    finally:
        for _, f in files:
            f.close()


if __name__ == "__main__":
    while True:
        print("\nOptions: [1] Query  [2] Upload Documents  [3] Exit")
        choice = input("Choose an option: ").strip()
        if choice == "1":
            q = input("Type a question: ")
            query_api(q)
        elif choice == "2":
            folder = input("Folder path to upload documents: ")
            upload_documents(folder)
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")
