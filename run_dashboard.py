import sys
import subprocess
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

def main():
    """Run the Streamlit dashboard"""
    try:
        dashboard_path = Path(__file__).parent / "src" / "dashboard" / "monitoring_app.py"
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ])
    except Exception as e:
        print(f"Failed to run dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
