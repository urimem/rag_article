import logging
from datetime import datetime

date_str = datetime.now().strftime("%d-%m-%Y")
log_file = f"log/run_{date_str}.log"
logger = logging.getLogger(__name__)
FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(filename=log_file, level=logging.INFO, format=FORMAT)

def log(text_to_append): 
	logger.info(text_to_append)
