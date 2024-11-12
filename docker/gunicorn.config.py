import os

workers = int(os.getenv("WEB_CONCURRENCY", 2))
threads = int(os.getenv("PYTHON_MAX_THREADS", 8))
max_requests = int(os.getenv("MAX_REQUESTS", 1000))
max_requests_jitter = int(os.getenv("MAX_REQUESTS_JITTER", 50))
