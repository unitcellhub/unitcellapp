import multiprocessing
multiprocessing.freeze_support()
import logging
from unitcellapp.index import production

# @NOTE:Multiprocessing needs to be imported first and then frozen.
# Otherwise, it causes issues with pyinstaller

# Set the logging level for the waitress server to INFO so that we can
# capture info about its load status. This is helpful for integration
# with Electron as we want to wait until the server is up before we
# load the window.
logger = logging.getLogger("waitress")
logger.setLevel(logging.INFO)

# Serve the application using waitress
production()
# serve(app.server, host="127.0.0.1", port=port, threads=threads)

