import logging

import unitcellapp.callbacks
from unitcellapp.app import app, port
from unitcellapp.layout import layout

logging.basicConfig()
logger = logging.getLogger("unitcellapp")
logger.setLevel(logging.WARNING)

# Set the app layout, which was defined in a submodule
app.title = "UnitcellApp"
app.layout = layout


# This is required when running as a deployed system using something like
# gunicorn
server = app.server


# Debug function
def debug() -> None:
    """Run unitcellapp in debug mode"""
    logger.setLevel(logging.DEBUG)
    app.run_server(debug=True, port=port)


# Run with waitress locally
def production() -> None:
    """Run unitcellapp in a production style WSGI"""
    # Determine the number of allowable threads to use
    import multiprocessing

    from waitress import serve

    threads = multiprocessing.cpu_count()
    if threads is None:
        threads = 8
    elif threads > 12:
        threads = 12

    # Serve the application using waitress
    serve(app.server, host="127.0.0.1", port=port, threads=threads)


# def pyinstaller() -> None:
#     """Create pyinstaller executable running WSGI interface"""
#     # The pyproject.toml is only capable of executing scripts that point to python
#     # module functions. The purpose of this function is to run pyinstaller to build
#     # a standalone executable.
#     # https://pyinstaller.org/en/stable/usage.html#running-pyinstaller-from-python-code
#     from pathlib import Path
#
#     import PyInstaller.__main__
#
#     PyInstaller.__main__.run(
#         [(Path(__file__).parent.parent.parent / Path("pyinstaller.spec")).as_posix()]
#     )
#

if __name__ == "__main__":
    # debug()
    production()
