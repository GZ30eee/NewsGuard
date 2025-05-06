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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
BACKEND_PORT = 5000
FRONTEND_PORT = 8501

def start_backend():
    """Start the Flask backend server"""
    logger.info("Starting backend server...")
    try:
        subprocess.run([sys.executable, "-m", "app.backend"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Backend server failed: {e}")
    except KeyboardInterrupt:
        logger.info("Backend server stopped by user")

def start_frontend():
    """Start the Streamlit frontend"""
    logger.info("Starting frontend server...")
    try:
        subprocess.run(["streamlit", "run", "app/frontend.py"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Frontend server failed: {e}")
    except KeyboardInterrupt:
        logger.info("Frontend server stopped by user")

def open_browser():
    """Open web browser after a delay"""
    time.sleep(5)  # Wait for servers to start
    url = f"http://localhost:{FRONTEND_PORT}"
    logger.info(f"Opening browser at {url}")
    webbrowser.open(url)

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("analysis_history", exist_ok=True)
    
    # Check if data files exist, if not create them
    if not os.path.exists("data/fake_news.csv"):
        logger.info("Creating sample fake news data file")
        with open("data/fake_news.csv", "w") as f:
            f.write("""title,text
"BREAKING: Cancer Cure Found in Common Fruit","BREAKING: Scientists confirm that drinking hot water mixed with lemon juice every morning can prevent cancer with 100% effectiveness. This revolutionary discovery has been suppressed by pharmaceutical companies for decades because it would eliminate the need for expensive cancer treatments. Share this information before it gets taken down!"
"Doctors Hate This One Simple Trick","Doctors don't want you to know this one simple trick that cures all diseases overnight. A local mom discovered this ancient remedy and now pharmaceutical companies are trying to silence her. Click here to learn the secret before it's banned!"
"SHOCKING: Government Hiding Alien Technology","SHOCKING: Government hiding alien technology that could solve all energy problems. Whistleblower reveals classified documents showing that the military has been reverse-engineering alien spacecraft since the 1950s. The technology could provide free energy to everyone, but oil companies are paying billions to keep it secret."
"Chocolate Diet Miracle","Studies show that eating chocolate every day makes you lose 10 pounds per week. Scientists at a prestigious university discovered that a special compound in chocolate activates your metabolism to burn fat 300% faster. The weight loss industry doesn't want you to know this simple fact!"
"Celebrity Reveals Secret to Immortality","Celebrity reveals secret to living to 150 years old - doctors hate him! This A-list actor hasn't aged in 20 years thanks to a strange fruit found only in a remote jungle. Hollywood elites have been using this secret for decades while keeping it from the public."
""")
    
    if not os.path.exists("data/real_news.csv"):
        logger.info("Creating sample real news data file")
        with open("data/real_news.csv", "w") as f:
            f.write("""title,text
"New Deep-Sea Fish Species Discovered","Scientists have discovered a new species of deep-sea fish that can withstand extreme pressure at depths of over 8,000 meters. The research team from the Marine Biology Institute published their findings in the journal Nature, detailing the unique physiological adaptations that allow these fish to survive in one of Earth's most hostile environments."
"Exercise May Reduce Cancer Risk","A recent study published in Nature suggests that regular exercise may reduce the risk of certain types of cancer by up to 20%. Researchers followed 10,000 participants over a 15-year period and found that those who exercised at least 150 minutes per week had significantly lower rates of colon and breast cancer compared to sedentary individuals."
"Federal Reserve Announces Interest Rate Increase","The Federal Reserve announced a 0.25% increase in interest rates following their quarterly meeting yesterday. The decision comes amid concerns about rising inflation and aims to stabilize the economy. Market analysts had widely anticipated this move, with most major stock indices showing minimal reaction to the news."
"Mars Rover Collects Promising Samples","NASA's Mars rover has collected rock samples that show evidence of ancient microbial life, according to preliminary analysis. The samples contain organic compounds and minerals typically formed in the presence of living organisms. Scientists caution that more testing is needed before making definitive claims about life on Mars."
"New Algorithm Improves Alzheimer's Detection","Researchers at Stanford University have developed a new algorithm that improves early detection of Alzheimer's disease. The AI system analyzes brain scans and can identify subtle changes up to six years before clinical symptoms appear. In clinical trials, the algorithm demonstrated 94% accuracy in predicting which patients would develop the disease."
""")
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend)
    backend_thread.daemon = True
    backend_thread.start()
    
    # Open browser after a delay
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start frontend (blocking call)
    start_frontend()