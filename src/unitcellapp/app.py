import dash
import dash_bootstrap_components as dbc
import os

# Setup Dash instance
port = 5030
basePrefix = "/"

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.4/dbc.min.css"
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.MINTY, dbc_css],
                requests_pathname_prefix=basePrefix)

# Define service for when running with gunicorn
# gunicorn dashboard.dashboard:server -b:5030
server = app.server
