from subprocess import Popen
import multiprocessing
from math import floor

def load_jupyter_server_extension(nbapp):
    """ Serve the dash-based flask app with gunicorn """
    # Regarding --worker-tmp-dir, see below reference:
    # https://docs.gunicorn.org/en/stable/faq.html#how-do-i-avoid-gunicorn-excessively-blocking-in-os-fchmod
    
    # Define the number of works for guincorn based on cpu count and the
    # rough rule of thumb provided in the gunicorn FAQ
    # https://docs.gunicorn.org/en/stable/design.html#how-many-workers
    # @TODO Right now, the use of modified global variables prevents the
    # use of more than one worker. To improve performance in the future, 
    # the global variable implementation should be updated to allow for
    # multiple workers.
    workers = 1
    threads = multiprocessing.cpu_count()

    
    Popen(["gunicorn", 
           "--bind=:5030",
           "--worker-tmp-dir=/dev/shm",
           f"--workers={workers}",
           f"--threads={threads}",
           "dashboard.index:server"])
