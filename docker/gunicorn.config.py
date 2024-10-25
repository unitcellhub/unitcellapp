import multiprocessing
import os

workers = int(os.getenv('WEB_CONCURRENCY', 2))
threads = int(os.getenv('PYTHON_MAX_THREADS', 8))
