import os

import dash
import dash_bootstrap_components as dbc

# Setup Dash instance
port = 5030
basePrefix = "/"

dbc_css = (
    "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.4/dbc.min.css"
)
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.MINTY, dbc_css],
    requests_pathname_prefix=basePrefix,
)

# Setup Google Analytics if specified in the environmental variables
gaID = os.getenv("GOOGLE_ANALYTICS_ID", False)
if gaID:
    gaHead = f"""
                <!-- Google tag (gtag.js) -->
                <script async src="https://www.googletagmanager.com/gtag/js?id={gaID}"></script>
                <script>
                window.dataLayer = window.dataLayer || [];
                function gtag(){{dataLayer.push(arguments);}}
                gtag('js', new Date());

                gtag('config', '{gaID}');
                </script>
    """
else:
    gaHead = ""

# Since Google Analytics needs to be added to the HTML head, the standards assets import can't
# be used because it gets loaded into the footer. So, he create a custom index_string that adds
# the analytics scripts to the head if and ID is found in the environmental variables.
app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        {gaHead}
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
"""


# Define service for when running with gunicorn
server = app.server
