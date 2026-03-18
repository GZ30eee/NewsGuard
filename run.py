import subprocess
import threading
import time
import webbrowser
import os
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Ensure all dependencies are installed"""
    logger.info("Verifying dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except Exception as e:
        logger.error(f"Failed to install dependencies: {e}")

def run_backend():
    """Start the Flask backend"""
    logger.info("Starting Flask backend...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    subprocess.run([sys.executable, "app/backend.py"], env=env)

def run_frontend():
    """Start the Streamlit frontend"""
    logger.info("Starting Streamlit frontend...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/frontend.py"])

if __name__ == "__main__":
    # 1. Setup Data
    from setup_data import setup_data
    setup_data()
    
    # 2. Install missing deps (Optional, usually good for first run)
    # install_dependencies() 

    # 3. Start Backend in background
    backend_proc = threading.Thread(target=run_backend, daemon=True)
    backend_proc.start()
    
    # 4. Wait slightly then open browser
    def open_link():
        time.sleep(5)
        webbrowser.open("http://localhost:8501")
    
    threading.Thread(target=open_link, daemon=True).start()
    
    # 5. Start Frontend (Main thread)
    run_frontend()