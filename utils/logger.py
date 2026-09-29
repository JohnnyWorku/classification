import logging
import os

# Create the log directory
os.makedirs("log", exist_ok=True)

# Configure the logger
logging.basicConfig(
    filename="log/log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filemode="a",
)

# Export a reusable logger
logger = logging.getLogger("classification")