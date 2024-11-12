import base64
import io
import json
import logging
import os
from copy import copy, deepcopy
from functools import partial
from pathlib import Path

import dash_bootstrap_components as dbc
import dill as pickle
import numpy as np
import pandas as pd
from dash import dcc, html
from PIL import Image
from unitcellengine.analysis.material import Ei, Gij, Ki, nuij
from unitcellengine.geometry.sdf import SDFGeometry

import unitcellapp
from unitcellapp.cache import CACHE, cacheLoad
from unitcellapp.options import (_DEFAULT_OPTIONS, _OPTIONS, NCUSTOM, OPTIONS,
                                 OPTIONS_NORMALIZE)

version = unitcellapp.__version__

# Setup logger
logger = logging.getLogger("unitcellapp")

# def augpow(x, exp=1):
#     return pow(x, exp)


BLANK_FIGURE = {"data": [], "layout": {}}


# Convert image data to encoded image string
def image2base64(mat):
    im = Image.fromarray(mat[0])
    buffer = io.BytesIO()
    im.save(buffer, format="png")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/png;base64, " + encoded_image
    return im_url


columns = list(_DEFAULT_OPTIONS.keys())
_DATA = {}
_SURROGATE = {}
_IMAGE = {}
# raise Exception("Forced recompute")
for cache in CACHE.glob("data_*.pkl"):
    _, form, unitcell = cache.stem.split("_")
    _data = cacheLoad("data", form, unitcell)
    try:
        _DATA[form][unitcell] = _data
        # In old version of the cache files, there are custom fields were
        # stored and we don't want to maintain them.
        _SURROGATE[form][unitcell] = partial(
            cacheLoad, kind="surrogates", form=form, unitcell=unitcell
        )
        _IMAGE[form][unitcell] = partial(
            cacheLoad, kind="images", form=form, unitcell=unitcell
        )
        # _SURROGATE[form][unitcell] = {
        #     k: v for k, v in tmp["surrogate"].items() if "custom" not in k
        # }
    except KeyError:
        # Initialize dictionaries as they don't exist yet
        _DATA[form] = {}
        _SURROGATE[form] = {}
        _IMAGE[form] = {}

        # Store data
        _DATA[form][unitcell] = _data

        # Store loader functions
        _SURROGATE[form][unitcell] = partial(
            cacheLoad, kind="surrogates", form=form, unitcell=unitcell
        )
        _IMAGE[form][unitcell] = partial(
            cacheLoad, kind="images", form=form, unitcell=unitcell
        )
if not _DATA or not _SURROGATE or not _IMAGE:
    raise IOError(
        f"No cached data files were found in {CACHE}. "
        "Download the UnitcellDB database from github.com/unitcellhub/unitcelldb "
        "and place the 'unitcelldb.h5' folder in the folder 'database' at the root "
        "of the UnitcellApp repository. Then, run the *createCache* method in the "
        "unitcellapp.cache module to generate the required cache files."
        "Run the *cacheCreate* method to generate the required cache files."
    )
logger.debug("Pre-processed cache file exists. Loaded cached data.")

# Split the graph form into truss and corrugations
_DATA_MOD = {"truss": {}, "corrugation": {}, "walledtpms": {}}
_SURROGATE_MOD = {"truss": {}, "corrugation": {}, "walledtpms": {}}
_IMAGE_MOD = {"truss": {}, "corrugation": {}, "walledtpms": {}}
for form, unitcells in _DATA.items():
    for unitcell in unitcells.keys():
        if "honeycomb" in unitcell:
            new = "corrugation"
        elif "graph" in form:
            new = "truss"
        else:
            new = form
        _DATA_MOD[new][unitcell] = _DATA[form][unitcell]
        _SURROGATE_MOD[new][unitcell] = _SURROGATE[form][unitcell]
        _IMAGE_MOD[new][unitcell] = _IMAGE[form][unitcell]
_DATA = _DATA_MOD
_SURROGATE = _SURROGATE_MOD
_IMAGE = _IMAGE_MOD
del _DATA_MOD
del _SURROGATE_MOD
del _IMAGE_MOD

# # Replace all custom _SURROGATE models with a modified definition
# for param in [f"custom{ind+1}" for ind in range(NCUSTOM)]:
#     for form, unitcells in _DATA.items():
#         for unitcell, values in unitcells.items():
#             # Store a different unity definition for the custom fields
#             _SURROGATE[form][unitcell][param] = {"model": ([], "nan"),
#                                                 "xscaler": None,
#                                                 "yscaler": None,
#                                                 "scores": None}
_SURROGATE_DEFAULT = ([], "nan")


def SURROGATE(scustom, form, unitcell):
    """Take in custom equations for evaluation with surrogate quantities

    Arguments
    ---------
    scustom: json string
        JSON list object of length NCUSTOM. Each element is a set of 2 entities.
        The 1st is a list of strings defining the custom equation variables and
        the 2nd is a string defining the custom equation.

    Returns
    -------
    dictionary of all surrogate models across all lattice forms and unitcell types
    """

    # Convert the input JSON data into python objects
    try:
        custom = json.loads(scustom)
        assert len(custom) == NCUSTOM, "Stored custom data has unexpected size."
    except:
        # If the input is empty, set as default parameters
        custom = [_SURROGATE_DEFAULT for _ in range(NCUSTOM)]

    # surrogate = copy(_SURROGATE)
    logger.info(f"Loading surrogate model for {unitcell} ({form})")
    logger.debug(_SURROGATE[form][unitcell])
    surrogate = _SURROGATE[form][unitcell]()
    logger.info("Surrogate model loaded")
    for param, model in zip([f"custom{ind+1}" for ind in range(NCUSTOM)], custom):
        # Store a different unity definition for the custom fields
        surrogate[param] = {
            "model": model,
            "xscaler": None,
            "yscaler": None,
            "scores": None,
        }

    return surrogate


# Go through and aggregate all of the data into a pandas dataframe,
# additionally keeping track of the bounds for each data metric
STANDARD = [col for col in columns if "custom" not in col]
NSTANDARD = len(STANDARD)
_BOUNDS_DEFAULT = [0.9, 1.1]
_BOUNDS = pd.DataFrame(np.ones((2, NSTANDARD)), columns=STANDARD)
_BOUNDS.iloc[0, :] *= 1e10
_BOUNDS.iloc[1, :] *= -1e10
IMAGES = _IMAGE
# IMAGES = {}
for k in _DATA.keys():
    # IMAGES[k] = {}
    for subk in _DATA[k].keys():
        # Store the images
        # IMAGES[k][subk] = [r[-1] for r in _DATA[k][subk]]

        # Store data
        _DATA[k][subk] = pd.DataFrame(
            np.array([r[:NSTANDARD] for r in _DATA[k][subk]]), columns=STANDARD
        )
        # _DATA[k][subk] = pd.DataFrame(np.array([r[:-1] for r in _DATA[k][subk]]),
        #                              columns=columns)

        # Stack up the column based bounds across all datasets
        submin = _DATA[k][subk].min()
        submax = _DATA[k][subk].max()
        _BOUNDS.iloc[0, :] = np.vstack((submin, _BOUNDS.iloc[0, :])).min(axis=0)
        _BOUNDS.iloc[1, :] = np.vstack((submax, _BOUNDS.iloc[1, :])).max(axis=0)

        # # Add in default custom values
        # for i in range(NCUSTOM):
        #     _DATA[k][subk][f'custom{i+1}'] = np.array([1]*len(_DATA[k][subk]))


def BOUNDS(sbounds):
    """Update bounds to include custom variable bounds

    Arguments
    ---------
    sbounds: JSON string
        List of upper and lower bounds for each custom variable. There should be
        NCUSTOM items in the list and, for each element, it should list the
        lower and upper bound for that custom variable.

    Returns
    -------
    Panda dataframe with the bounds for all QOI (including custom QOI)

    """

    # Convert the input JSON data into python objects
    try:
        custom = json.loads(sbounds)
        assert len(custom) == NCUSTOM, "Stored custom bounds data has unexpected size."
    except:
        # If the input is empty, set as default parameters
        custom = [_BOUNDS_DEFAULT for _ in range(NCUSTOM)]

    # Create full data frame with all bounds
    bounds = _BOUNDS.copy()
    for i, minmax in enumerate(custom):
        bounds[f"custom{i+1}"] = minmax

    return bounds


_DATA_DEFAULT = pd.DataFrame(
    np.array([[1] * NCUSTOM for r in _DATA[k][subk]]),
    columns=[col for col in columns if "custom" in col],
).to_dict()


def DATA(sdata):
    """Update data dataframe with custom data

    Arguments
    ---------
    sdata: JSON string
        Dictionary (form) of dictionaries (unitcell type) of dictionaries
        (dataframe in dictionary form)

    Returns
    -------
    Dictionary of dictionaries containing pandas dataframes (same format at _DATA)

    """

    # Convert the input JSON data into a standard python object
    try:
        custom = json.loads(sdata)
        assert (
            custom.keys() == _DATA.keys()
        ), "Custom stored data has the incorrect from keys."
        for form in _DATA.keys():
            assert (
                custom[form].keys() == _DATA[form].keys()
            ), "Custom stored data has the informed unitcell keys."
    except:
        custom = {
            form: {unitcell: _DATA_DEFAULT for unitcell in _DATA[form].keys()}
            for form in _DATA.keys()
        }

    # Combine default and custom values into one dataframe
    # Note that we need a deep copy here rather than a shallow
    # copy since there are mutable global objects within this
    # dictionary. There was a significant memory leak when
    # a shallow copy was used.
    # data = deepcopy(_DATA)
    data = {}
    for form in _DATA.keys():
        data[form] = {}
        for unitcell, df in _DATA[form].items():
            data[form][unitcell] = pd.concat(
                (df.copy(), pd.DataFrame(custom[form][unitcell]))
            )
    #
    return data


CARD_STYLE = {"width": "33%"}
# _UNITCELLS = [f"{u} (graph)" for u in _DATA['graph'].keys()] +\
#              [f"{u} (walled tpms)" for u in _DATA['walledtpms'].keys()]
# _UNITCELLS = [f"{u} (graph)" for u in SDFGeometry._GRAPH_UNITCELLS] +\
#              [f"{u} (walled tpms)" for u in SDFGeometry._WALLED_TPMS_UNITCELLS]


# General method for creating slider-based quantity-of-interest (QOI)
# Filters. This allows the user to reduce the viseable data to a subset
# of the data set.
def _qoiFilter(qoi):
    info = _DEFAULT_OPTIONS[qoi]

    # Pull out the bound and round them to nice numbers based on the
    # range of the data
    vmin, vmax = BOUNDS(None)[qoi]
    if np.isclose(vmin, vmax):
        # If the bounds are the same, offset them slightly to prevent a
        # divide by zero issue
        vmin *= 0.9
        vmax *= 1.1
    scaling = 10 ** np.round(np.log10(vmax - vmin) - 2)
    vmin = np.floor(vmin / scaling) * scaling
    vmax = np.ceil(vmax / scaling) * scaling

    # # Create an index quantity to aid in matching custom quantities
    # index = 100
    # if "custom" in qoi:
    #     index = int(qoi[6:])

    # Create a slider for this QOI filter
    # Getting the slider to vertically align with other elements was
    # challenging. The current implemention used a pretty hacky use of
    # the translate transform.
    step = (vmax - vmin) / 100
    children = [
        html.Label(
            info["name"],
            title=info["info"],
            id={"type": "slider-qoiFilter-label", "qoi": qoi},
        ),
        html.Div(
            [
                dcc.Input(
                    id={"type": "slider-qoiFilter-lower", "qoi": qoi},
                    min=vmin,
                    max=vmax,
                    value=vmin,
                    debounce=True,
                    style={"width": "4rem"},
                    persistence=False,
                ),
                html.Div(
                    dcc.RangeSlider(
                        id={"type": "slider-qoiFilter", "qoi": qoi},
                        min=vmin,
                        max=vmax,
                        step=step,
                        value=[vmin, vmax],
                        tooltip={"always_visible": False, "placement": "bottom"},
                        allowCross=False,
                        persistence=False,
                        marks=None,
                    ),
                    style={
                        "align-self": "flex-end",
                        "flex": "1 0",
                        "transform": "translateY(25%)",
                    },
                ),
                dcc.Input(
                    id={"type": "slider-qoiFilter-upper", "qoi": qoi},
                    min=vmin,
                    max=vmax,
                    value=vmax,
                    debounce=True,
                    style={"width": "4rem"},
                    persistence=False,
                ),
            ],
            style={
                "width": "50%",
                "display": "flex",
                "justify-content": "start",
                "align-items": "center",
            },
        ),
    ]
    return html.Div(
        id=f"div-qoiFilter-{qoi}", children=children, style={"display": "none"}
    )


def _customQoi(ind):
    """Create custom QOI input boxes"""

    title = html.H5(
        f"Custom QOI {ind}:", title=_DEFAULT_OPTIONS[f"custom{ind}"]["info"]
    )
    label = html.Div(
        dbc.Input(
            id={"type": "custom-qoi-label", "qoi": f"custom{ind}"},
            debounce=True,
            type="text",
            placeholder=f"Label",
            persistence=False,
        ),
        title=("Label used to identify the custom QOI " "throughout the unitcellapp."),
    )
    equation = html.Div(
        dbc.Input(
            id={"type": "custom-qoi-eqn", "qoi": f"custom{ind}"},
            debounce=True,
            type="text",
            placeholder=f"Equation",
            style={"width": "100%"},
            persistence=False,
        ),
        title=(
            "Equation that defines the custom QOI. "
            "The equation should use Python syntax "
            "and can reference multiple QOIs ("
            "simply hover over the QOI of interest "
            "and note the 'variable name') and Numpy "
            "functions such as 'sin', 'exp', etc'. "
            "Note that QOI references are case sensitive."
        ),
    )

    row = dbc.Row(
        [dbc.Col(title, width=2), dbc.Col(label, width=2), dbc.Col(equation)],
        justify="start",
    )

    return row


def _createSelected(column):
    """Create an interactive figure element for a selected unit cell"""

    out = [
        html.H3(f"Selected {column}"),
        dcc.Loading(
            id=f"selected{column}-loading",
            children=[
                html.Div(
                    id=f"selected{column}-viz",
                    #    style=dict(height="200px")
                ),
                dcc.Store(id=f"selected{column}-vizPrevious"),
            ],
            parent_className="loading_wrapper",
        ),
    ]

    return out


def _createCard(column):
    """Create the elements for a unit cell score card column"""

    DIM_SLIDER_PARAMS = dict(
        min=1,
        max=5,
        step=0.05,
        value=1,
        tooltip={"always_visible": True, "placement": "right"},
        persistence=False,
        marks={v: f"{v}" for v in [1, 1.25, 1.5, 2, 3, 4, 5]},
    )
    NUM_SLIDER_PARAMS = dict(
        min=1,
        max=4,
        step=1,
        value=1,
        tooltip={"always_visible": True, "placement": "right"},
        persistence=False,
    )

    options = []
    for form, kinds in _DATA.items():
        for kind, values in kinds.items():
            # Pull out a reference image for the geometry
            try:
                # Find a moderatly low density geometry with a unit aspect ratio
                inds = np.logical_and.reduce(
                    (
                        values["xyAR"] > 0.99,
                        values["xyAR"] < 1.01,
                        values["yzAR"] > 0.99,
                        values["yzAR"] < 1.01,
                        values["relativeDensity"] > 0.05,
                        values["relativeDensity"] < 0.15,
                    )
                )
                image = IMAGES[form][kind]()[np.argmax(inds)]
            except:
                # If there were issues getting an image, just use the
                # first in the list
                image = IMAGES[form][kind]()[0]

            # Define a human readable name for the unitcell form
            if form == "walledtpms":
                ref = "walled TPMS"
            else:
                ref = form

            # Create a custom label that includes an image of the unitcell
            value = f"{kind} ({ref})"
            label = html.Div(
                [
                    html.Img(src=image, height=30),
                    html.Div(value, style={"padding-left": 10}),
                ],
                style={
                    "display": "flex",
                    "align-items": "center",
                    "justify-content": "left",
                },
            )
            # label = html.Div(value)
            options.append(dict(label=label, value=value))

    out = [
        html.H3(f"Case {column}"),
        dcc.Dropdown(
            options=options,
            id={"type": "card-unitcell", "index": column},
            value=None,
            persistence=False,
        ),
        html.Br(),
        dbc.Checklist(
            id={"type": "card-ARlock", "index": column},
            options=[{"label": "Lock aspect ratio", "value": "lock"}],
            value=[],
            switch=True,
        ),
        dcc.Store(id={"type": "card-AR", "index": column}),
        html.Label("Length", title=_DEFAULT_OPTIONS["length"]["info"]),
        dcc.Slider(id={"type": "card-L", "index": column}, **DIM_SLIDER_PARAMS),
        html.Label("Width", title=_DEFAULT_OPTIONS["width"]["info"]),
        dcc.Slider(id={"type": "card-W", "index": column}, **DIM_SLIDER_PARAMS),
        html.Label("Height", title=_DEFAULT_OPTIONS["height"]["info"]),
        dcc.Slider(id={"type": "card-H", "index": column}, **DIM_SLIDER_PARAMS),
        html.Label("Thickness", title=_DEFAULT_OPTIONS["thickness"]["info"]),
        dcc.Slider(
            id={"type": "card-T", "index": column},
            min=0.01,
            max=0.5,
            step=0.01,
            value=0.3,
            tooltip={"always_visible": True, "placement": "right"},
            persistence=False,
            marks=None,
        ),
        dcc.ConfirmDialog(
            id={"type": "card-T-issue", "index": column},
            message=(
                "The current thickness value for "
                f"design Case {column} "
                "is outside the bounds of existing "
                "precomputed metrics. All QOI "
                "values are therefore inaccurate "
                "and should not be used. For "
                "accurate predictions, move "
                "the thickness slider between or "
                "on existing markers on the "
                "slider bar."
            ),
        ),
        #    html.Label("Smoothing"),
        #    dcc.Slider(id={'type': 'card-R', 'index': column}, min=0., max=0.5, step=0.01,
        #               value=0.3, tooltip={'always_visible': True,
        #                                   'placement': 'right'},
        #               persistence=False, marks=None),
        html.Label(
            "Length-wise number of cells",
            title=(
                "This is for visualization purposes only. "
                "Define the number of displayed unit cells "
                "in the x direction."
            ),
        ),
        dcc.Slider(id={"type": "card-nx", "index": column}, **NUM_SLIDER_PARAMS),
        html.Label(
            "Width-wise number of cells",
            title=(
                "This is for visualization purposes only. "
                "Define the number of displayed unit cells "
                "in the y direction."
            ),
        ),
        dcc.Slider(id={"type": "card-ny", "index": column}, **NUM_SLIDER_PARAMS),
        html.Label(
            "Height-wise number of cells",
            title=(
                "This is for visualization purposes only. "
                "Define the number of displayed unit cells "
                "in the z direction."
            ),
        ),
        dcc.Slider(id={"type": "card-nz", "index": column}, **NUM_SLIDER_PARAMS),
        html.Label(
            "Resolution",
            title=(
                "This is for visualization purposes only. "
                "It defines the mesh resolution relative to "
                "the thickness parameter for the design. "
                "The smaller the value, the better the "
                "quality of the rendering, but at the cost "
                "of increased rendering time. In general, "
                "the default value should provide sufficient "
                "resolution."
            ),
        ),
        dcc.Slider(
            id={"type": "card-res", "index": column},
            min=float(os.getenv("RESOLUTION_LOWER", 0.25)),
            max=0.5,
            step=0.01,
            value=0.5,
            tooltip={"always_visible": True, "placement": "right"},
            persistence=False,
            marks=None,
        ),
        #    dcc.Interval(id=f"card{column}-interval", interval=10,
        #                 disabled=True),
        dcc.Loading(
            id=f"card{column}-viz-loading",
            children=html.Div(id={"type": "card-viz", "index": column}),
        ),
        html.Br(),
        dcc.Loading(
            id=f"card{column}-props-loading",
            children=html.Div(id={"type": "card-props", "index": column}),
        ),
    ]

    return out


aboutText = """

The Lattice Design Tool is a comprehensive design tool.
It leverages a database of more than 10,000 simulated point designs.
Provides a 

"""
about = [
    html.Br(),
    dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4("Design exploration", className="card-title"),
                            html.P(
                                (
                                    "Explore a wide range of unit cell "
                                    "geometry types and parameterizations "
                                    "via pre-computed and custom design metrics "
                                    "presented in stacked Ashby plots."
                                ),
                                className="card-text",
                            ),
                            html.Img(
                                src="assets/explore.png",
                                style={"align": "center", "width": "100%"},
                            ),
                        ]
                    ),
                    style={"height": "100%"},
                ),
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4("Simulation-based", className="card-title"),
                            html.P(
                                (
                                    "Built upon the well established "
                                    "homogenization multiscale "
                                    "modeling approach, allowing for "
                                    "large scale marcroscopic representations "
                                    "while maintaining mesoscale details."
                                ),
                                className="card-text",
                            ),
                            html.Img(
                                src="assets/homogenization.svg",
                                style={
                                    "display": "block",
                                    "margin-left": "auto",
                                    "margin-right": "auto",
                                },
                            ),
                        ]
                    ),
                    style={"height": "100%"},
                ),
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4(
                                "Enhanced by Machine Learning", className="card-title"
                            ),
                            html.P(
                                (
                                    "Over 10,000 lattice geometries feed into "
                                    "a Gaussian Process Regression-based approach, "
                                    "enabling performance predictions of arbitrary "
                                    "lattice geometries."
                                ),
                                className="card-text",
                            ),
                            html.Img(
                                src="assets/gpr.svg",
                                style={"align": "center", "width": "100%"},
                            ),
                        ]
                    ),
                    style={"height": "100%"},
                )
            ),
        ]
    ),
]
# about = dbc.Card(
#     dbc.CardBody(
#         [
#             html.H2("About", className="card-title"),
#             html.P(aboutText,
#                    className="card-text")
#         ]
#     ),
#     className="mt-3",
# )

FAQ_RESOURCES = """
A good place to start is "Cellular Solids: Structure and Properties" by
Gibson and Ashby. This book provides insight into how lattices work,
practical methods to model and design them for engineering applications,
and in-depth case studies. The primary topic not covered in this book is
homogenization theory. See "What is homogenization theory?" below for
more information and references.
"""

FAQ_HOMOGENIZATION = """
Homogenization theory is a multiscale modeling technique initially
developed in the mid-1900s to model structures with underlying
periodicity. Some of the most common applications for homogenization
theory include modeling of composites and cellular/lattice structures.
In both cases, the large length-scale separation between the macroscopic
size of engineer structures and their small-scale
(mesoscopic/microscopic) substructure make conventional modeling
techniques impractical; for example, explicitly modeling each ligament
of a lattice structure with ligaments on the order of 1 mm in size over
an entire engineering structure on the order of 1000 mm, results in an
extremely large finite element mesh (likely on the order of 100 million
to 1 billion elements in size). Although recent advances in computation
power and advanced linear solvers (such as implicit solvers) have made
solving some of these problems tractable, large scale problems like this
are still extremely cumbersome and challenging to work with. Multi-scale
modeling techniques like homogenization help bridge this gap by
leveraging known structures within the problem space (such as
periodicity in the case of homogenization) to reduce modeling
complexity.

In its simpiliest form, homogenization theory represents the large-scale
structure using effective material properties (such as mechanical
stiffness) and a mapping of macroscopic quantities (such as macroscopic
stress) to the local unit cell (fundamental repeating unit of the
structure) level (such as local level stresses). In the case of single
material lattice structures, the effective properties of the macroscopic
structure correspond to an orthotropic material with properties $E_x$,
$E_y$, $E_z$, $G_{xy}$, $G_{yz}$, $G_{xz}$, $\\nu_{xy}$,
$\\nu_{yz}$, and $\\nu_{xz}$, where *E*, *G*, and $\\nu$ are elastic
modulus, shear modulus, and Poisson's ratio, respectively. The large
scale structure is modeled using these effective properties. Local
unit cell properties (displacement, stress, strain, temperature) are then
calculated at areas of interest through a linear mapping.

The detailed theory behind homogenization is beyond the scope of this
FAQ; however, here is a list of a few references for those that are
interested to learn more. 
- Pinho-da-Cruz, J., Oliveira, J. A., & Teixeira-Dias, F. (2009).
"Asymptotic homogenisation in linear elasticity. Part I: Mathematical
formulation and finite element modelling." Computational Materials
Science, 45(4), 1073-1080.
https://doi.org/10.1016/J.COMMATSCI.2009.02.025 
- Oliveira, J. A., Pinho-da-Cruz, J., & Teixeira-Dias, F. (2009).
"Asymptotic homogenisation in linear elasticity. Part II: Finite element
procedures and multiscale applications." Computational Materials
Science, 45(4), 1081-1096.
https://doi.org/10.1016/j.commatsci.2009.01.027 
- Dong, G., Tang, Y., & Zhao, Y. F. (2019). "A 149 Line Homogenization
Code for Three-Dimensional Cellular Materials Written in MATLAB."
Journal of Engineering Materials and Technology, Transactions of the
ASME, 141(1), 1-11. https://doi.org/10.1115/1.4040555 
- Andreassen, E., & Andreasen, C. S. (2014). "How to determine composite
material properties using numerical homogenization." Computational
Materials Science, 83, 488–495.
https://doi.org/10.1016/j.commatsci.2013.09.006


"""

FAQ_ACCURACY = """
The fundamental assumptions of homogenization theory are length scale
separation and underlying periodicity: such that the lattice is periodic
and at least 100x smaller than the macroscopic scale of the structure,
homogenization is quite accurate. If the unit cell length scale is
between 10-100x smaller than the macroscopic scale, the stiffness
results are still likely pretty accurate, but stresses less so. Another
factor in all of this is the accuracy of the manufacturing process. For
example, conventional metallic honeycombs have relatively poor geometric
tolerances (with the exception of foil thickness) and additively
manufactured lattices likely have local asperities or are overbuild at
ligament/plate nodes. In these cases, the stiffnesses are again likely
pretty accurate, but the stresses are more likely to deviate from
predictions. Ultimately, these data are intended to help an engineer
downselect a design appropriate for a given application rather than
provide a highly accurate set of metrics. **All performance should be
independently verified in the context of the application to ensure the
desired performance**.
"""

FAQ_NOW_WHAT = """
The primary purpose of this tool is to help a designer sift through the
complicated lattice design space and downselect candidate geometries for
a given application. From there, 
- The homogenized properties should be modeled in the context of the
given application to verify the unit cell's applicability. 
- Beyond basic stiffness/conductance assessments, local unit cell
quantities such as stress should be examined to verify that they are a
reasonable order of magnitude. Note that the validity of these local
quantities strongly depends on the length-scale separation of the
geometry (i.e., how much smaller the lattice unit cell is than the large
scale structure), how periodic the lattice is, and how close to the
lattice boundary the quantity is. In general, they quantities should be
thought of a quick "hand calc" assessment that needs further refinement.
- If the design still appears to be applicable, a more detailed design
verification should be conducted. The specifics of this detailed
verification depends on the problem size (i.e., is it possible to
explicitly model the structure with the finite element method) and the
risk posture of the application. In some cases, beam/shell-based model
representations might be acceptable while in other cases a full 3D mesh
and might be required. For low risk posture applications, testing will
also likely be required.

"""

FAQ_GENERATION = """
All presented data are the result of simulation. To explore the
available design space, geometric parameter sweeps were conducted on a
set of common unit cell geometries (approximately 30 in total). To probe
the effects of unit cell aspect ratios and relative density, a range of
unit cell lengths, widths, heights, and ligament/shell thicknesses were
varied. For simplicity, this was a brute force approach, some times
resulting in redundant data points; for example, unit cells with dimensions (length x width x
height) 1x2x3, 2x1x3, and 3x2x1 were all sampled, even though they are
simple rotations of each other (for most unit cells). Additionally, the
geometric parameters were chosen to span a reasonable portion of viable
geometric designs (with a focus on low relative densities); however,
this search was not exhaustive.

The geometric definitions were defined using an implicit sign-distance
approach similar to tools like nTopology. This implementation was chosen
over conventional CAD-based approaches due to its improved robustness
(for example, at high relative densities) and its ability to capture
complex geometric forms such as TPMS structures. The Python library
[sdf](https://github.com/fogleman/sdf) was used as the primary backend
engine and expanded upon to generate beam, shell, and TPMS unit cells.  

A Python-based in-house finite element solver was developed to solve the
homogenization problem. This may sound like "reinventing the wheel,"
however, 
- few commerial finite element solvers have the capability to
explicitly solve the full homogenization problem (it is sometimes possible 
via 2-stage loading); 
- license restriction make this large scale parameter study infeasible
(over 10,000 designs were sampled) 
- automated post processing if often challenging. 

Due to the complex geometric forms often found in lattice unit cells (in
particular, TPMS geometries)
- tetrahedral-based meshers often fail
- it is usually challenging to generate the periodic mesh required for
homogenization. 

A voxel-based meshing approach was therefore implemented. The
downside of this approach is that the surface geometry has a "stair step"
representation rather than the real smooth surface. To better capture
stiffness and stresses on the surface of the geometry, a density-based
approach (comparable to those used in Topology Optimization algorithms)
was therefore incorporated over these surface elements. It should be
noted that this methodology captures stiffness with a reasonable
accuracy, but less so for stresses.


"""

FAQ_TPMS = """

Triply Periodic Minimum Surfaces (TPMS) define a set of geometries that - have
translation periodicity in all three orthonormal directions - are constructed
from minimal surfaces. Minimal surfaces are locally area-minimizing. This
results in surfaces that have zero mean curvature and are correspondingly smooth
in nature. A good example of a minimal surface in nature is a soap film. Walled
TPMS structures are simply thickened TPMS. The currently implemented
TPMS lattices correspond to walled-TPMS geometries; that is, the nominal TPMS
surface is thickened to creat thin walls.

Although a subset of TPMS structures can be explicitly defined, a more flexible
approach is to define them by a level-set function such that the nominal surface of the
structure is defined by the function Φ(x, y, z) = 0.  For the exact definitions used in this
tool, select a unit cell form from the below dropdown list (where L, W, and H
are respectively the unitcell length, width, and height).

"""

# @TODO: This should be generalized to read the SDF definitions
FAQ_TPMS_EQUATIONS = {
    "gyroid": "sin(2πx/L)cos(2πy/W) + sin(2πy/W)cos(2πz/H) + sin(2πz/H)cos(2πx/L)",
    "schwarz": "cos(2πx/L) + cos(2πy/W) + cos(2πz/H)",
    "iwp": (
        "2 [cos(2πx/L)cos(2πy/W) + cos(2πy/W)cos(2πz/H) + cos(2πz/H)cos(2πx/L)]"
        " - (cos(4πx/L) + cos(4πy/W) + cos(4πz/H))"
    ),
    "diamond": (
        "sin(2πx/L)sin(2πy/W)sin(2πz/H) + "
        "sin(2πx/L)cos(2πy/W)cos(2πz/H) + "
        "cos(2πx/L)sin(2πy/W)cos(2πz/H) + "
        "cos(2πx/L)cos(2πy/W)sin(2πz/H)"
    ),
    "lidinoid": (
        "sin(4πx/L)cos(2πy/W)sin(2πz/H) + "
        "sin(4πy/W)cos(2πz/H)sin(2πx/L) + "
        "sin(4πz/H)cos(2πx/L)sin(2πy/W) - "
        "cos(4πx/L)cos(4πy/W) - "
        "cos(4πy/W)cos(4πz/H) - "
        "cos(4πz/H)cos(4πx/L) + 0.3"
    ),
    "splitp": (
        "1.1 [sin(4πx/L)*sin(2πz/H)*cos(πy/W) + "
        "sin(4πy/W)*sin(2πx/L)*cos(2πz/H) + "
        "sin(4πz/H)*sin(πy/W)*cos(2πx/L)] - "
        "0.2 [cos(4πx/L)*cos(4πy/W) + "
        "cos(4πy/W)*cos(4πz/H) + "
        "cos(4πz/H)*cos(4πx/L)] - "
        "0.4 [cos(4πx/L)+cos(4πy/W)+cos(4πz/H)]"
    ),
    "neovius": "3 [cos(2πx/L)+cos(2πy/W)+cos(2πz/H)]+4cos(2πx/L)cos(2πy/Wcos(2πz/H)",
}

FAQ_WALLEDTPMS_OPTIONS = {}
form = "walledtpms"
for kind, values in _DATA[form].items():
    # Pull out a reference image for the geometry
    try:
        # Find a moderatly low density geometry with a unit aspect ratio
        inds = np.logical_and.reduce(
            (
                values["xyAR"] > 0.99,
                values["xyAR"] < 1.01,
                values["yzAR"] > 0.99,
                values["yzAR"] < 1.01,
                values["relativeDensity"] > 0.05,
                values["relativeDensity"] < 0.15,
            )
        )
        image = IMAGES[form][kind]()[np.argmax(inds)]
    except:
        # If there were issues getting an image, just use the
        # first in the list
        image = IMAGES[form][kind]()[0]

    # Create a custom label that includes an image of the unitcell
    value = f"{kind}"
    label = html.Div(
        [html.Img(src=image, height=30), html.Div(value, style={"padding-left": 10})],
        style={"display": "flex", "align-items": "center", "justify-content": "left"},
    )
    # label = html.Div(value)
    FAQ_WALLEDTPMS_OPTIONS[kind] = dict(
        equation=FAQ_TPMS_EQUATIONS[kind.lower()], label=dict(label=label, value=value)
    )

FAQ_GRAPH = """

Truss and corrugation unit cells define a set of geometries that are constructed
from beams and plates, respectively. Each geometric form is defined by a set of nodes 
(or vertices) and the corresponding beam/plate connectivity relative to 
these nodes. For the exact definitions used in this tool, select a
unit cell form from the below dropdown list (where X, Y, and Z are normalized
coordinates within the unitcell). 

"""
FAQ_GRAPH_OPTIONS = []
forms = ["truss", "corrugation"]
for form in forms:
    for kind, values in _DATA[form].items():
        # Pull out a reference image for the geometry
        try:
            # Find a moderatly low density geometry with a unit aspect ratio
            inds = np.logical_and.reduce(
                (
                    values["xyAR"] > 0.99,
                    values["xyAR"] < 1.01,
                    values["yzAR"] > 0.99,
                    values["yzAR"] < 1.01,
                    values["relativeDensity"] > 0.05,
                    values["relativeDensity"] < 0.15,
                )
            )
            image = IMAGES[form][kind]()[np.argmax(inds)]
        except:
            # If there were issues getting an image, just use the
            # first in the list
            image = IMAGES[form][kind]()[0]

        # Create a custom label that includes an image of the unitcell
        value = f"{kind}"
        label = html.Div(
            [
                html.Img(src=image, height=30),
                html.Div(value, style={"padding-left": 10}),
            ],
            style={
                "display": "flex",
                "align-items": "center",
                "justify-content": "left",
            },
        )
        # label = html.Div(value)
        FAQ_GRAPH_OPTIONS.append(dict(label=label, value=value))


FAQ_STRESSES = """
Homogenization theory provides a well defined mapping between
macroscopic and local unit-cell stresses, which was calculated for all
sampled unit cell geometries; however, as stress is a field quantity
that depends on the specific macroscopic stress state of a system (both
normal and shear stresses), it wasn't possible to provide the full
stress amplification data in the design tool. To help bridge this
gap and give a sense of general stress amplification for a given design,
the peak von Mises stresses within a unit cell is provided for 6x
independent load cases: unit uniaxial loading in the x, y, and z 
directions and unit shear loading in the xy, yz, and xz directions.
In the case of linear analysis, these results can be scaled and
superimposed to predict the worst case local stress state for an arbitrary
macroscopic stress. For example, if the predicted stress state from a
homogenized structure is \[$\\sigma_{xx}^{macro}$, $\\sigma_{yy}^{macro}$,
$\\sigma_{zz}^{macro}$, $\\sigma_{xy}^{macro}$, $\\sigma_{yz}^{macro}$,
$\\sigma_{xz}^{macro}$\] and the max local *amplification* stresses for a
given unit cell are \[$\\sigma_{xx}^{max local}$, $\\sigma_{yy}^{max local}$,
$\\sigma_{zz}^{max local}$, $\\sigma_{xy}^{max local}$, $\\sigma_{yz}^{max local}$,
$\\sigma_{xz}^{max local}$\], then the enveloping worst case local
stresses are simply \[$\\sigma_{xx}^{macro} \cdot \\sigma_{xx}^{max local}$,
$\\sigma_{yy}^{macro} \cdot \\sigma_{yy}^{max local}$,
$\\sigma_{zz}^{macro} \cdot \\sigma_{zz}^{max local}$,
$\\sigma_{xy}^{macro} \cdot \\sigma_{xy}^{max local}$,
$\\sigma_{yz}^{macro} \cdot \\sigma_{yz}^{max local}$,
$\\sigma_{xz}^{macro} \cdot \\sigma_{xz}^{max local}$\].

A few notes:
- This local stress methodology can easily be implemented into most FEM
post processing tools whereby a new field is calculated based on the
homogenized stress field and the unit cell stress amplifications.
- This method should be a conservative representation of the stress
state as the max stress amplification terms are not location consistent.
That is, the max stress amplification of the unit x loading likely
doesn't occur in the same location within the unit cell as the unit y
loading. 
- All stress information is currently related to von Mises stress, which
is only relevant for ductile materials. In the case of brittle
materials, the max principle stresses are required, which are not currently
unavailable in the tool (although are possible to implement if
requested). 
"""

FAQ_NORMALIZATION = """
By default, all Quantities of Interest (QOI) are presented as normalized
quantities. For example, all stiffness quantities are normalized by the
Young's modulus of the base material. If the reported stiffness value is
0.2, the predicted stiffness for an Aluminum lattice is $0.2 \cdot 68 =
13.6$ GPa (noting that the Young's modulus of Aluminum is 68 GPa). This
allows for general presentation of information that is both agnostic to
the base material (with some limitations) and the preferred unit system
(i.e., metric vs imperial). For details regarding the normalization
method for each QOI, simply hover over the QOI when selecting it in the
Explore tab or in any of the tabulated results.

To Explore and Compare absolute quantities, simply create a Custom QOI
on the Explore Tab that multiplies the absolute normalizing factor with
the QOI of interest. For example, a Custom QOI for $E_{xx}$ (with units
of GPa) for Aluminum lattices would be defined as "E1*68", where E1 is
the variable name reference for $E_{xx}$ in UnitcellApp. To determine
the variable name for the QOI of interest, simply hover over the QOI in
the pulldown menu on the Explore tab and look for the "variable name" at
the end of the text.

*NOTE*: The one caveat to the normalization methodology is that the base
material is assumed to have a Poisson's ratio of 0.3. This is valid for
most metallic materials; however, is less so for polymerics and
ceramics.

"""

FAQ_LIMITATIONS = """
To make sound judgements based on the information presented in this
tool, it is important to understand its underlying limitations. 
There are a few key assumptions that form the backbone of
this tool:
- All mechanical properties are based on linear elasticity. Non-linear
phenomenon such as plasticity and buckling are not currently included.
- All mechanical properties are based on a material with a Poisson's
ratio of 0.3. Since most metallic materials have a Poisson's ratio within
a few percent of this value, this assumption holds for most metallic
applications. In general, this assumption is less valid for polymers,
which tend to have a higher Poisson's ratio, especially in the rubbery
state. 
- Homogenization theory relies on an underlying periodicity to the
lattice structure.
- Homogenization theory relies on length scale separation between the
underlying lattice structure and the macroscopic geometry. A good
general rule of thumb is to have at least 5 unit cells through the 
thickness of the macroscale geometry.
- There are no manufacturing constraints incorporated into the tool.
Sound engineering judgement is required to ensure the selected design
not only meets application requirements, but also manufacturing
constraints. In the case of additively manufactured lattices, there are
two primary constraints: geometric overhangs and powder removal.

"""

FAQ_IN_WORK = """
Section in work...
"""

FAQ_CITE = """
Please reference UnitcellApp in any research report, journal
or publication that requires citation of any author's work. Your
recognition of this resource is important for acquiring funding for the
future improvements and development. The minimal content of a citation
should include:

Watkins, R. "UnitcellApp: a lattice design tool." Jet Propulsion Laboratory.

"""

FAQ_REQUEST_FEATURE = """
All feedback regarding the tool usability and requested features should
be emailed to Ryan Watkins at <ryan.t.watkins@jpl.nasa.gov>. This tool
does not have active funding support; so, depending on
the size and scope of the requested features, funding may be required to
implement new functionality.

There are some feature additions currently under consideration:
- Addition of linear buckling QOI
- Export an STL of a given unit cell design
- Export homogenized properties to a Nastran input deck format
- Topology optimization of a unit cell given specified design
constraints 

"""

faq = dbc.Card(
    dbc.CardBody(
        [
            html.H2("FAQ", className="card-title"),
            html.Br(),
            html.Div(
                [
                    html.H3("Theory"),
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                [
                                    dcc.Markdown(FAQ_RESOURCES),
                                    html.Iframe(
                                        src="https://read.amazon.com/kp/card?asin=B00JOK9I6O&preview=inline&linkCode=kpe&ref_=cm_sw_r_kb_dp_YR4F08MPFR7EZJ2DKFHR",
                                        width="336",
                                        height="550",
                                        style={"max-width": "100%"},
                                    ),
                                ],
                                title="Where can I find further resources for lattice design and modeling?",
                            ),
                            dbc.AccordionItem(
                                [dcc.Markdown(FAQ_HOMOGENIZATION, mathjax=True)],
                                title="What is homogenization theory?",
                            ),
                            dbc.AccordionItem(
                                [dcc.Markdown(FAQ_ACCURACY)],
                                title="How accurate are the data points?",
                            ),
                            dbc.AccordionItem(
                                [dcc.Markdown(FAQ_LIMITATIONS)],
                                title="What are the limitations of this tool?",
                            ),
                            dbc.AccordionItem(
                                [dcc.Markdown(FAQ_GENERATION)],
                                title="How were these data created?",
                            ),
                            dbc.AccordionItem(
                                [
                                    dcc.Markdown(FAQ_TPMS),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    dcc.Dropdown(
                                                        [
                                                            v["label"]
                                                            for v in FAQ_WALLEDTPMS_OPTIONS.values()
                                                        ],
                                                        value=None,
                                                        id="faq-walledtpms",
                                                    ),
                                                ],
                                                style=dict(width="50%"),
                                            ),
                                            html.Div(id="faq-walledtpms-definition"),
                                        ]
                                    ),
                                ],
                                title="What is a TPMS unit cell?",
                            ),
                            dbc.AccordionItem(
                                [
                                    dcc.Markdown(FAQ_GRAPH),
                                    html.Div(
                                        [
                                            dcc.Dropdown(
                                                FAQ_GRAPH_OPTIONS,
                                                value=None,
                                                id="faq-graph",
                                            ),
                                            html.Div(id="faq-graph-definition"),
                                        ],
                                        style=dict(width="50%"),
                                    ),
                                ],
                                title="What is are truss/corrugation unit cell?",
                            ),
                        ],
                        start_collapsed=True,
                    ),
                    html.Br(),
                    html.H3("Usage"),
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                [dcc.Markdown(FAQ_NOW_WHAT)],
                                title="I've selected a design. Now what?",
                            ),
                            dbc.AccordionItem(
                                [dcc.Markdown(FAQ_STRESSES, mathjax=True)],
                                title="How do I calculate stresses?",
                            ),
                            dbc.AccordionItem(
                                [dcc.Markdown(FAQ_NORMALIZATION, mathjax=True)],
                                title="What are the units?",
                            ),
                            dbc.AccordionItem(
                                [dcc.Markdown(FAQ_CITE)],
                                title="How do I cite this tool?",
                            ),
                            dbc.AccordionItem(
                                [dcc.Markdown(FAQ_REQUEST_FEATURE)],
                                title="How do I request a new feature?",
                            ),
                        ],
                        start_collapsed=True,
                    ),
                ]
            ),
        ]
    ),
    className="mt-3",
)

EXAMPLES_PREAMBLE = """

Designing lattice structures can often be nuanced and daunting. They are
generally anisotopic (although UnitcellApp only has data for orthotopic lattices) and are often used for their
non-linear properties, such as energy absorbing crushables. To help orient the
user, a series of examples are presented for common use cases. These
examples, however, are not all encompassing. On the contrary, the whole goal of
***UnitcellApp*** is to provide lattice data in a way such that new applications can
be found. Therefore, treat these examples as a primer to enable you to expand
beyond the status quo.

"""

EXAMPLES_UNITAR = """

In some instances, a given application may require unit aspect ratio
unit cells; that is, unit cells with the same length, width, and height.

- In the *Explore* tab, this can be accomplished by the addition of two
filters: *X/Y aspect ratio* and *Y/Z aspect ratio*. Simply set both
filter ranges to 0.99 and 1.01. See the attached configuration file for
reference. 
- In the *Compare* tab, this can be accomplished by activating the *Lock
aspect ratio* toggle for the relevant geometry case.

"""

EXAMPLES_ENERGY = r"""

### Background
Lattice structures are commonly used in energy absorption applications due to
their unique crushing behavior. Their usage can be found in parcel packaging
(such as foam inserts), sports helmets, and the Apollo lunar landing legs to
name a few examples. Two primary properties make lattices ideal energy
absorbers:

- Their hierarchical structure often leads to localized crushing behavior. This
  localized crushing behavior can typically be sustained over a large range of
  stroke (often on the order of 50-75%) and with low structural stiffness. Since
  energy is defined by the area under a stress-strain curve, lattices can absorb
  a large amount of energy while limiting loading on a sensitive/delicate item
  that needs to be protected.
- The crushing behavior found in lattices can dissipate kinetic energy through
  mechanisms such as visco-elasticity, plasticity, and/or fracture. This
  minimizes rebound loading and other undesirable dynamic effects.

The macroscopic behavior of a crushing lattice can be broken into three regimes
as shown below: a short linear elastic region, a long stress plateau, and a
final increase in stiffness as the structure densifies (that is, the structure
has been crushed to the point that it starts to act as a solid rather than a
lattice). From a design perspective, the plateau stress should be selected to be
lower than the stress it takes to break the sensitive/delicate item being
protected. Based on this plateau stress, the densification strain should be
higher than the crush strain required to absorb the input kinetic energy.

![Characteristic lattice crushing behavior](assets/honeycombCrushing.svg)

Due to the prevalence of existing crushable lattice applications, several
methodologies already exist to characterize the efficiency of crushable energy
absorbers. The most common metrics are the Janssen and cushion factors. In both
cases, full stress-strain (either simulated or experimental) are required to
define these metrics. As these data are beyond the scope of **UnitcellApp**, a
slightly different methodology is outlined in the below example. For more
details on more conventional crushable lattice characterization, see Chapter 8
in [Cellular
Solids](https://www.cambridge.org/core/books/cellular-solids/BC25789552BAA8E3CAD5E1D105612AB5).
Additionally, another good reference is [HexWeb® Honeycomb Energy Absorption
Systems: Design
Data](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwitzaSOyuSCAxWGP0QIHfK-D0cQFnoECBMQAQ&url=https%3A%2F%2Fwww.physicsforums.com%2Fattachments%2Fhexwebhoneycombenergyabsorptionbrochure-pdf.16488%2F&usg=AOvVaw0ciPZUz809hPoaEEldCrZs&opi=89978449). 

### Egg drop design problem

Let's design a lattice crushable for the classic egg drop challenge. The primary
objective is to protect a raw egg from breaking when dropped from 2 stories
above the ground. 

- To focus the design on the crushable rather than aerodynamics, neglect the
  effects of drag. This effectually prevents the addition of high-drag features
  to the design (such as parachutes).
- The egg has a mass of 60 g.
- One story is equivalent to 3.3 m. So, the egg must be able to survive a drop
  of 6.6 m. In the absence of drag, potential energy is fully converted to
  kinetic energy. As a point of reference, for just the egg alone, $m_e gh =
  (0.06)(9.81)(6.6) = 3.88$ J. 
- Assume that the minimum egg diameter is 40 mm. That is, the worst-case
  projected impact area is 1257 mm<sup>2</sup>.
- Assume that the breaking strength of the egg is 0.18 MPa and that the
  crushable will evenly distribute the impact loading onto the egg.
- Assume we will manufacture the crushable out of PETG plastic using a desktop 3D
  printer. 
    - Yield stress - 50 MPa
    - Material density - 1230 kg/m<sup>3</sup>



### Step 1: Find lattices with viable crush strengths 
To begin with, we know that the egg breaks under 0.18 MPa of stress. So, we want
our crushable to have a crush strength less than this value. To incorporate some
margin of safety into our design, let's target a crushable that, at most,
crushes at 0.16 MPa of stress. This is the first metric we can examine in
**UnitcellApp**. 

Although the crushing behavior found in lattices is highly nonlinear -- material
nonlinearity, buckling, and contact -- and **UnitcellApp** properties are based on
linear elasticity, we can still make some educated decisions to help downselect
the lattice design domain. The crush stress of a lattice is primarily driven by
either yielding- or buckling-induced plastic collapse. **UnitcellApp** doesn't
currently have any buckling metrics (although it will in the future), but it
does have the ability to predict when a lattice will start to yield. To do this,
we will utilize the "Max stress amplification" QOI. *Note that this is a rather
drastic assumption and we therefore need to be careful in how we interpret the
subsequent data; in particular, we need to reduce our investigation to "bending"
dominated lattices as they are less prone to buckling-induced crushing. The full
details of this are beyond the scope of this example, but a 1st order spot check
is to determine if the loading results in bending or uniaxial stress in the
members of the lattice.*

The relationship between max stress in the lattice and the macroscopic stress
exerted on the egg is related by 

$$σ_{lattice} = A σ_{egg}$$

where *A* is the stress amplification factor for the lattice loaded in a
particular direction.  Based on this relationship, and noting that yielding
starts slightly prior to the stress plateau, the plasticity-induced crush
strength of a lattice is approximately

$$σ_{crush} ≈ C σ_{yield}/A$$

where $C$ is an empirical constant that relates the onset of crushing in a
lattice to the onset of yielding in the material (which is generally around 2-4)
and $σ_{yield}$ is the yield stress of the constituent material in the lattice.

### Step 2: Estimated the densification strain

Like crush strength, densification strain is not a QOI in **UnitcellApp**; so, we'll
need to extrapolate relevant **UnitcellApp** QOI to inform our downselection
process. Intuitively, the more void space in a lattice, the more space the
lattice has to crush. For conventional foams, it has been empirically observed
that

$$ε_{d} ≈ 1 - 1.4 ρ^*$$

where $ρ^*$ is the relative density of the foam (see Eq 5.22 in [Cellular
Solids](https://www.cambridge.org/core/books/cellular-solids/BC25789552BAA8E3CAD5E1D105612AB5)
book highlighted in the FAQ section). Although the lattices in
**UnitcellApp** generally aren't foams, this will still work as a good first-order
estimate of densification strain.

### Step 3: Estimate the crushing energy density

The total energy absorbed by a lattice corresponds to the area under the
crushing force-displacement curve:

$$W = \int F dδ$$

where *δ* is crush displacement and *F* is the applied load. Ignoring the
initial linear elastic response and assuming a flat crush plateau,

$$W_{max} ≈ (σ_{crush} A_0)(ε_{d} L_0) = A_0 L_0 (σ_{crush}  ε_{d})$$

where $A_0$ is the crush area and $L_0$ is the nominal length of the crushable
(in the crush direction). As $A_0 L_0$ corresponds to a volume, $σ_{crush}  ε_{d}$
can be thought of as the maximum energy density for the lattice. To achieve a
low-mass design, we're interested in maximizing this quantity within the bounds
of the design problem.

### Step 4: Downselecting candidate lattices

Putting Steps 1-3 together, let's downselect viable lattice candidates for our
design problem in the *Explore* tab of **UnitcellApp**. To help facilitate the
investigation, we'll define three *Custom QOI* (select "Define custom QOI" in
the QOI section):


| Custom QOI number | Label | Equation |
|-------------------|-------|----------|
| 1                 | Estimated crush strength (MPa) | 3*50/vonMisesWorst33 |
| 2                 | Estimated densification strain (mm/mm) | 1-1.4*relativeDensity |
| 3                 | Estimated energy density (MPa mm/mm) | custom1*custom2 |

A few notes on the above table:

- The *Label* input is only used when labeling axes for a custom quantity. They
  therefore aren't required; however, they help improve the readability and
  minimize user errors.
- In the *Equation* input, you can define the relationship (in Python syntax)
  for your custom QOI. These equations can reference existing QOI (such as the
  *relativeDensity* reference in Custom QOI 2) along with previously defined
  custom QOI (such as the *custom1* and *custom2* references in Custom QOI 3).
  To determine the appropriate variable name for a QOI, find it in the
  *Quantities of Interest (QOI)* dropdown, hover over it and note down the
  *variable name* at the end of the text.
- These definitions can be downloaded by pressing the below *Download config*
  button and loading it on the *Explore* tab.
- The "Estimated crush strength" custom QOI is defined with respect to loading
  the unit cell "z" direction. Since lattices are orthotropic, this only
  quantifies a subset of possible behaviors. For a more general investigation,
  all three loading directions could be investigated.

With the custom QOI defined, select "Relative density" and "Estimated energy
density (MPa mm/mm)" under the *Quantities of interest (QOI)* dropdown and move
down to the plot at the bottom of the screen. We're most of the way there,
however, this plot currently shows designs with no restriction on crush
strength, which is a critical aspect of our design. If we scroll up to the
*Filters* section, we can select the "Estimated crush strength (MPa)" QOI in the
dropdown box and then adjust the corresponding slider to limit the maximum crush
strength to 0.16 MPa (which is are maximum allowable crush strength). Now, if we
go back to the plot, we only see viable lattices. From here, we're interested in
finding lattices with a low relative density (i.e., low mass) and high energy
density. Using these two guiding principles, it looks like either a 5x5x5
Truncated Octahedron (3.4% dense) or a 3x3x5 Body Centered Cubic (3.7% dense)
lattice might be good candidates for this design problem. Note that the
Truncated Cube meets these criteria, but the vertical members make them
susceptible to buckling, which isn't accounted for in our estimate for
crush strength; so, it is likely they will have a lower crush strength due to
elastic buckling (although it is hard to say without further investigation).

### Step 5: Size the lattice

With our lattice candidates selected, we need to size the lattice for our
specific application. Our lattice needs to be able to absorb the full potential
energy from dropping the egg/lattice system:

$$PE=(m_{egg}+m_{lattice})gH = PE_{egg} + m_{lattice}gH$$

where *g=9.81 m/s<sup>2</sup>* is the acceleration due to gravity, *H=6.6* m is our drop
height, *m<sub>egg</sub>=60* g is our egg mass, and *m<sub>lattice</sub>* is the mass of our
crushable lattice as defined by

$$m_{lattice}=ρ^*ρ_s A_0 L_0$$

where $ρ_s =1230$ kg/m<sup>3</sup> is the density of PETG. Assuming conservation
of energy (*PE=W*),

$$PE=W = A_0 L_0 (σ_{crush}  ε_{crush})$$

noting that we select $ε_{crush} < ε_{d}$ to include some margin of safety in
our design (as even a little bit of densification is likely to break our egg).
Here, designs are estimated to densify at around 90% strain, so we'll select
$ε_{crush}=70$%. The only unknown is $L_0$, which can be solved for:

$$L_0 = \frac{PE_{egg}}{A_0(σ_{crush}ε_{crush}-ρ^*ρ_s g H)}$$

For our two candidate lattices,

| Type | Dimensions | Relative density (%) | Crush strength (MPa) | $L_0$ (mm) |
|------|-----------|----------------------|--------------------|-------|
| Truncated Octahedron | 5x5x5 | 3.4 | 0.149 | 30.4 |
| Body Centered Cubic | 3x3x5 | 3.7 | 0.153 | 29.6 |

Lastly, the unit cell geometries are non-dimensional forms that need to be
converted to absolute units. To do this, we need to select the smallest possible
unit cell that we can 3D print with good quality. For both lattices, the
non-dimensional ligament thickness is 0.3. Assuming we're printing with a 0.4 mm
nozzle (which is the most common nozzle size on desktop 3D printers), the
minimum strut we can print is 0.8 mm. The absolute scaling factor is therefore
$0.8/0.3=2.67$ and the corresponding cell sizes should be 13.35 x 13.35 x 13.35
mm and 8x8x13.35 mm for the Truncated Octahedron and Body Centered Cubic
lattices, respectively. Since we can fit more Body Centered Cubic lattices
throughout the crush area, that is likely the ideal candidate to start with.

### Step 5: Testing and simulation

This lattice selection and sizing process relied on several simplifications.
These design candidates should therefore be considered a starting point rather
than the final design. From here, simulation or testing should be used to refine
the design for the given application.

"""

EXAMPLES_THERMAL = """

### Background

Lattice structures have emerged as a promising alternative to conventional fluid heat
exchanger designs due to their unique combination of properties that enhance
heat transfer efficiency. Unlike conventional heat
exchangers that rely on solid fins or corrugated channels to increase surface
area, lattice structures offer a more intricate and interconnected network of
voids, creating a vast expanse of heat transfer surfaces within a compact
volume. This increased design flexibility has the potential to:

- increase the surface area to volume ratio (SA/V);
- tune the direction of thermal conductivity;
- increase turbulence by roughening the surfaces; and
- minimize pressure drop through controlled fluid channels.

### Fluid heat exchanger design problem

Let's design a fluid heat exchanger to manage the heat generated by an
electronics board.  


"""

examples = dbc.Card(
    dbc.CardBody(
        [
            html.H2("Examples", className="card-title"),
            html.Br(),
            dcc.Markdown(EXAMPLES_PREAMBLE),
            html.Div(
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
                            [
                                dcc.Markdown(FAQ_IN_WORK),
                                # html.Div(
                                #     [dbc.Button("Download config", color="secondary",
                                #                 id={"type": "button-example",
                                #                     "example": "sandwich"}),
                                #      dcc.Download(id={"type": "download-example",
                                #                      "example": "sandwich"},
                                #                   type='application/json')]
                                # )
                            ],
                            title="High bending stiffness sandwich panel",
                        ),
                        dbc.AccordionItem(
                            [
                                dcc.Markdown(FAQ_IN_WORK),
                                # html.Div(
                                #     [dbc.Button("Download config", color="secondary",
                                #                 id={"type": "button-example",
                                #                     "example": "isolator"}),
                                #      dcc.Download(id={"type": "download-example",
                                #                      "example": "isolator"},
                                #                   type='application/json')]
                                # )
                            ],
                            title="Thermal isolator",
                        ),
                        dbc.AccordionItem(
                            [
                                dcc.Markdown(
                                    EXAMPLES_ENERGY,
                                    mathjax=True,
                                    dangerously_allow_html=True,
                                ),
                                html.Div(
                                    [
                                        dbc.Button(
                                            "Download config",
                                            color="secondary",
                                            id={
                                                "type": "button-example",
                                                "example": "crushable",
                                            },
                                        ),
                                        dcc.Download(
                                            id={
                                                "type": "download-example",
                                                "example": "crushable",
                                            },
                                            type="application/json",
                                        ),
                                    ]
                                ),
                            ],
                            title="Energy absorber",
                        ),
                        dbc.AccordionItem(
                            [
                                dcc.Markdown(EXAMPLES_UNITAR),
                                html.Div(
                                    [
                                        dbc.Button(
                                            "Download config",
                                            color="secondary",
                                            id={
                                                "type": "button-example",
                                                "example": "unitar",
                                            },
                                        ),
                                        dcc.Download(
                                            id={
                                                "type": "download-example",
                                                "example": "unitar",
                                            },
                                            type="application/json",
                                        ),
                                    ]
                                ),
                            ],
                            title="Only unit aspect ratio geometries",
                        ),
                    ],
                    start_collapsed=True,
                )
            ),
        ]
    ),
    className="mt-3",
)

explore = dbc.Card(
    dbc.CardBody(
        [
            html.H2(
                "Explore the design space",
                className="card-title",
                title=(
                    "This section of the design tool is intendend "
                    "to be the first step in the design process. "
                    "Here, relevant quantities of interest for "
                    "a given application are defined, filtered, "
                    "and plotted to give a global sense of "
                    "their behaviors and interactions. From "
                    "here, a set of canditate designs are "
                    "identified and then refined in the "
                    "'Compare' section of the tool."
                ),
            ),
            html.Br(),
            html.H3(
                children="Quantities of interest (QOI)",
                title=(
                    "QOI define relevant geometric and performance "
                    "metrics for a given lattice design. These "
                    "metrics range from relative density, to "
                    "elastic stiffness, to thermal conductance."
                ),
            ),
            dcc.Dropdown(
                options=OPTIONS(_DEFAULT_OPTIONS),
                value=["relativeDensity"],
                multi=True,
                id="qoi",
                persistence=False,
            ),
            html.Br(),
            dcc.Store("equations"),
            dcc.Store("curves"),
            dcc.Store("customSurrogates"),
            dcc.Store("customData"),
            dcc.Store("customBounds"),
            dcc.Store("customOptions"),
            # dcc.Loading(html.Div([_customQoi(ind) for ind in range(1, NCUSTOM+1)]),
            #             type="default",
            #             parent_className='loading_wrapper'
            # ),
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            dcc.Loading(
                                html.Div(
                                    [_customQoi(ind) for ind in range(1, NCUSTOM + 1)]
                                ),
                                type="default",
                                parent_className="loading_wrapper",
                            )
                        ],
                        title="Define custom QOI",
                    ),
                ],
                start_collapsed=True,
            ),
            html.Br(),
            html.H3(
                children="Filters",
                title=(
                    "To more easily nagivate the design space "
                    "and to restrict the investigation to "
                    "feasible quantifies for a given application, "
                    "set lower and upper bounds on relevant "
                    "QOI."
                ),
            ),
            dcc.Dropdown(
                options=[
                    o
                    for o in OPTIONS(_DEFAULT_OPTIONS)
                    if o["value"] not in ["relativeDensity", "Emax"]
                ],
                value=[],
                multi=True,
                id="extraFilters",
                persistence=False,
            ),
            html.Br(),
            dcc.Loading(
                html.Div(
                    id="filters",
                    children=[_qoiFilter(qoi) for qoi in columns],
                ),
                type="default",
                parent_className="loading_wrapper",
            ),
            html.Br(),
            html.H3(
                children="Normalization",
                title=(
                    "Sometimes it is useful to normalize all of "
                    "the QOI by another QOI to provide greater "
                    "insight into fundamental trends. A "
                    "good example of this is to normalize by "
                    "relative density to better understand the "
                    "mass efficiency of designs."
                ),
            ),
            dcc.Dropdown(
                options=OPTIONS_NORMALIZE(_DEFAULT_OPTIONS),
                value=0,
                id="normalization",
                persistence=False,
                clearable=False,
            ),
            html.Br(),
            html.H3(
                children="Unit cell types",
                title=(
                    "Lattice unitcell forms can be categorized "
                    "into different types based on the "
                    "form of their geometric definition. "
                    "Sometimes it can be useful to filter "
                    "an investigation based on these "
                    "different unitcell types."
                ),
            ),
            dcc.Dropdown(
                options=[
                    dict(
                        label="Truss",
                        value="truss",
                        title="Subset of unit cell geometries that can be defines by a set of connected coordinates defining beams (ex. octet truss).",
                    ),
                    dict(
                        label="Corrugation",
                        value="corrugation",
                        title="Subset of unit cell geometries that can be defines by a set of connected coordinates defining directionally oriented plates (ex hexagonal honeycomb).",
                    ),
                    dict(
                        label="Walled TPMS",
                        value="walledtpms",
                        title="Subset of unit cell geometries that can be defined by Triply Periodic Minimal Surfaces (https://en.wikipedia.org/wiki/Triply_periodic_minimal_surface). Common geometries include the Gyroid and Lidinoid cells.",
                    ),
                ],
                value=["truss", "corrugation", "walledtpms"],
                multi=True,
                id="forms",
                persistence=False,
            ),
            html.Br(),
            html.H3(
                children="Save/load state",
                title=(
                    "To aid in subsequent investigations, "
                    "the current state of the 'Explore' tab "
                    "can be saved and then later reloaded "
                    "during a different session. Note that "
                    "loading a configuration file will "
                    "completely overwrite the existing state."
                ),
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                dbc.Button(
                                    "Save",
                                    id="save-custom",
                                    style={
                                        "width": "100%",
                                        "height": "60px",
                                        "lineHeight": "60px",
                                        "borderWidth": "1px",
                                        "borderStyle": "solid",
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                    },
                                    color="light",
                                ),
                                dcc.Download(
                                    id="download-custom", type="application/json"
                                ),
                            ]
                        )
                    ),
                    dbc.Col(
                        dcc.Upload(
                            ["Drag and Drop or ", html.A("Select a File")],
                            style={
                                "width": "100%",
                                "height": "60px",
                                "lineHeight": "60px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "5px",
                                "textAlign": "center",
                            },
                            accept="application/json",
                            max_size=1e6,
                            id="load-custom",
                        )
                    ),
                ]
            ),
            dcc.ConfirmDialog(
                id="load-custom-issue",
                message=(
                    "Unable to load custom QOI definitions. "
                    "Check the file to ensure it was generated "
                    "by the unitcellapp.save feature."
                ),
            ),
            dcc.Store("loadedConfig"),
            dcc.Store("loadedFilters"),
            dcc.Store("doubleClick"),
            dcc.Store("selectedData"),
            html.Br(),
            html.H3(
                children="QOI plots",
                title=(
                    "QOI relevant to a given application can be "
                    "selected under the 'Quantities of Interest "
                    "(QOI)' section and plotted as Ashby style "
                    "plots below. If more than 2x QOI are "
                    "selected, a pyramid of plots is generated "
                    "with shared axes. Selecting points within "
                    "a plot will highlight the corresponding "
                    "points in the other plots. A 'score card' will "
                    "also be generated, presenting all of its "
                    "QOI. A maximum of 3x 'score cards' can be "
                    "generated. All subsequent selected points "
                    "will not be shown. "
                    "To deselect all of the selected points, click the 'Reset "
                    "selection' button."
                ),
            ),
            dbc.Button(
                "Reset selection",
                id="resetSelection",
                color="primary",
                style={"align": "center"},
            ),
            # https://community.plotly.com/t/show-loading-spinner-over-graph/46992/2
            # @TODO: Would like to have the graph partially greyed out during loading
            dcc.Loading(
                children=[
                    html.Div(
                        id="wrapperGraphAshbyPlots",
                        children=dcc.Graph(
                            id="graphAshbyPlots",
                            figure=BLANK_FIGURE,
                            clear_on_unhover=True,
                        ),
                    )
                ],
                type="default",
                parent_className="loading_wrapper",
            ),
            dcc.Tooltip(id="graphAshbyPlotsTooltip"),
            html.Br(),
            html.Div(id="selectedComparison"),
            html.Br(),
            # dbc.Row(
            #     # style={"width": "100%", "height": "400px"},
            #     children=[dbc.Col(id='selected1', children=_createSelected(1)),
            #             dbc.Col(id='selected2', children=_createSelected(2)),
            #             dbc.Col(id='selected3', children=_createSelected(3))],
            # ),
        ]
    ),
    className="mt-3",
)

COMPARE_UNCERTAINTIES = """
A Machine Learning framework is used here to interrogate
arbitrary design configurations. The majority of the time, this
implementation is robust and accurate; however, for certain geometries
and QOI, the quality of the model breaks down. In these instances, the
QOI are highlighted in red and estimated uncertainty bounds are
presented as a ± quantity.
"""

COMPARE_MARKERS = """
For the *Length*, *Width*, *Height*, and *Thickness* sliders, the markers
indicate datapoints explicitly solved for during the parameter study
and used in the Machine Learning training. For slider points that lie
between markers, the Machine Learning predictions are likely accurate,
while they fall off sharply outside of them. **Pay particular attention
with the *Thickness* slider as certain unit cells were only evaluated at one
value. These Machine Learning estimates should therefore only be used at
the marker location. If all of the calculated QOI are highlighted red, 
one of the sliders is likely outside of the parameter study points.** 
"""

compare = dbc.Card(
    dbc.CardBody(
        [
            html.H2(
                "Compare point designs",
                className="card-title",
                title=(
                    "This section is the second stage in "
                    "the design process. After exploring "
                    "the global trends in the 'Explore' "
                    "section and identifing candidate lattice "
                    "designs, they can be further understood "
                    "and refined here. Compare up to 3x design "
                    "concepts at once, tweaking their geometric "
                    "definitions to better fit your application."
                ),
            ),
            html.Br(),
            dbc.Card(
                [
                    dbc.CardHeader(html.H5("Notes")),
                    dbc.CardBody(
                        dbc.Accordion(
                            [
                                dbc.AccordionItem(
                                    [dcc.Markdown(COMPARE_UNCERTAINTIES)],
                                    title="Machine Learning uncertainties",
                                ),
                                dbc.AccordionItem(
                                    [dcc.Markdown(COMPARE_MARKERS, mathjax=True)],
                                    title="Slider markers",
                                ),
                            ],
                            start_collapsed=True,
                        )
                    ),
                ],
                # className="w-50",
            ),
            html.Br(),
            dbc.Row(
                # style={"width": "100%", "height": "400px"},
                children=[
                    dbc.Col(id="card1", children=_createCard(1)),
                    dbc.Col(id="card2", children=_createCard(2)),
                    dbc.Col(id="card3", children=_createCard(3)),
                ],
            ),
        ]
    ),
    className="mt-3",
)


layout = dbc.Container(
    [
        html.Br(),
        dbc.Row(
            children=[
                dbc.Col(
                    children=html.Img(
                        src="assets/logo.svg",
                        style={"align": "center", "height": "80px"},
                    ),
                    width=dict(size=1),
                    align="center",
                ),
                dbc.Col(
                    children=html.H1(
                        children="UnitcellApp: a lattice design tool",
                        className="card-title",
                    ),
                    width=dict(size=11),
                ),
            ],
            align="center",
            className="g-0",
        ),
        html.Br(),
        dbc.Tabs(
            [
                dbc.Tab(about, label="Home"),
                dbc.Tab(explore, label="Explore"),
                dbc.Tab(compare, label="Compare"),
                dbc.Tab(examples, label="Examples"),
                dbc.Tab(faq, label="FAQ"),
            ]
        ),
        html.Br(),
        html.Hr(),
        html.Footer(
            dbc.Stack(
                [
                    html.Div(
                        [
                            "© 2024 ",
                            html.A(
                                "UnitcellHub", href="https://www.github.com/unitcellhub"
                            ),
                            " team. All rights reserved. ",
                        ]
                    ),
                    html.Div("", className="mx-auto"),
                    html.Div([f"{'' if 'Unknown' in version else 'v' + version}"]),
                ],
                direction="horizontal",
                gap=3,
            ),
        ),
        html.Br(),
    ],
    className="dbc",
)
