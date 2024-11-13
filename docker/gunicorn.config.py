import os

# https://pythonspeed.com/articles/gunicorn-in-docker/
workers = int(os.getenv("WEB_CONCURRENCY", 2))
threads = int(os.getenv("NUM_THREADS", 4))
max_requests = int(os.getenv("MAX_REQUESTS", 1000))
max_requests_jitter = int(os.getenv("MAX_REQUESTS_JITTER", 50))
worker_tmp_dir = "/dev/shm"
