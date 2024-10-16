import multiprocessing

multiprocessing.freeze_support()
from requests import head
from waitress import serve
from dashboard.app import app, port
from dashboard.layout import layout
import dashboard.callbacks
import logging
import socket

# from urllib import request, parse
import requests
import getpass
from datetime import datetime

# Setup logging
# logging.basicConfig()
# logger = logging.getLogger("unitcell")
# logger.setLevel(logging.ERROR)

# Set the logging level for the waitress server to INFO so that we can
# capture info about its load status. This is helpful for integration
# with Electron as we want to wait until the server is up before we
# load the window.
logger = logging.getLogger("waitress")
logger.setLevel(logging.INFO)

# Set the app layout, which was defined in a submodule
app.title = "UnitcellApp"
app.layout = layout

# This is required when running as a deployed system using something like
# gunicorn
server = app.server

# Determine the number of allowable threads to use
# @NOTE: Due to an issue with pyinstaller and the multiprocessing module
# (see https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html#multi-processing
# for more details), the os module is used here instead.
threads = multiprocessing.cpu_count()
if threads is None:
    threads = 8
elif threads > 12:
    threads = 12

# Serve the application using waitress
serve(app.server, host="127.0.0.1", port=port, threads=threads)
