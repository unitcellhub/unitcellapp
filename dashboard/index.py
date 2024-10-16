import logging
logging.basicConfig()
logger = logging.getLogger("unitcell")
logger.setLevel(logging.DEBUG)

from dashboard.app import app, port
from dashboard.layout import layout
import dashboard.callbacks



# Set the app layout, which was defined in a submodule
app.title = "UnitcellApp"
app.layout = layout


# This is required when running as a deployed system using something like 
# gunicorn
server = app.server

if __name__ == '__main__':
    
    app.run_server(debug=True, port=port)