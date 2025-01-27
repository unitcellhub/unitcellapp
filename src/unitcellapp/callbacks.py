import base64
import json
import logging
import re
import textwrap
from io import BytesIO
from itertools import cycle
from pathlib import Path
from tokenize import NAME, STRING, tokenize, untokenize

import dash
import dash_bootstrap_components as dbc
import dash_vtk
import numpy as np
from dash import ALL, MATCH, Input, Output, State, html
from dash_vtk.utils import to_mesh_state
from memory_profiler import profile
from plotly import graph_objects as go
from plotly import subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.validators.scatter.marker import SymbolValidator
from unitcellengine.geometry.sdf import GRAPH_DEF, SDFGeometry

from unitcellapp.app import app
from unitcellapp.customeval import customeval
from unitcellapp.layout import (_BOUNDS_DEFAULT, _DATA, _SURROGATE_DEFAULT,
                                BLANK_FIGURE, BOUNDS, DATA,
                                FAQ_WALLEDTPMS_OPTIONS, IMAGES, SURROGATE,
                                columns)
from unitcellapp.options import (_DEFAULT_CUSTOM, NCUSTOM, OPTIONS,
                                 OPTIONS_NORMALIZE, _options)

# Precompile a regular expression to parse custom variables in a custom equation
REGEX_CUSTOM_VAR = re.compile("(custom\d+)")

# Setup logger
logger = logging.getLogger("unitcellapp")


# Define function to parse triggered property ids
def parsePropID(value):
    """Parse triggered property id information

    Arguments
    ---------
    value: str
        'prop_id' value from triggered item. See example below for form.

    Return
    ------
    Property definition: dict
    Property type: str

    Example form:
    [
        {
            'prop_id': '{"index":0,"type":"dynamic-dropdown"}.value',
            'value': 'NYC'
        }
    ]

    """
    chunks = value.split(".")
    prop = json.loads(".".join(chunks[:-1]))
    kind = chunks[-1]
    return prop, kind


# app.clientside_callback(
#     ClientsideFunction(namespace="clientside", function_name="trigger_hover"),
#     Output("dummy", "data-hover"),
#     [Input("graphAshbyPlots", "clickData")],
# )
CONFIG_FILE = Path(__file__).parent.parent / Path("config.json")
RAW_SYMBOLS = SymbolValidator().values
_values = []
for i in range(0, len(RAW_SYMBOLS), 3):
    name = RAW_SYMBOLS[i + 2]
    # Skip select symbols
    if (
        "dot" in name
        or "triangle" in name
        or "arrow" in name
        or "thin" in name
        or "y-" in name
        or "asterisk" in name
        or "hash" in name
        or "line" in name
    ):
        continue
    _values.append(name)
# logger.debug(_values)
MARKER_SYMBOLS = _values


# @profile
def _createSubCardViz(unitcell, L, W, H, T, R, form, nx, ny, nz, res):
    """Create a score card vizualization for a given design"""

    # Create a 3D model of the geometry
    try:
        form = form.replace(" ", "").lower()
        # Load the geometry
        logger.debug(f"Rendering geometry")
        design = SDFGeometry(
            unitcell, L, W, H, thickness=T, radius=R, elementSize=res, form=form
        )
        logger.debug(f"Geometry definition: {design}")
        # geom = design.visualizeVTK(int(nx), int(ny), int(nz))
        geom = design.visualizeVTK(1, 1, 1)

        # Tile the geometry
        logger.debug(f"Tiling geometry: [{nx}, {ny}, {nz}]")
        for j, n in enumerate([int(nx), int(ny), int(nz)]):
            tile = []
            for i in range(1, n):
                direction = i * np.eye(3)[j, :]
                translate = direction * np.array([L, W, H]) * design.DIMENSION
                tile.append(geom.copy(deep=True).translate(translate))
            geom = geom.merge(tile, merge_points=False)

        # geom = clip.extract_geometry().triangulate().decimate_pro(0.75)
        # sized = geom.compute_cell_sizes()
        # relativeDensity = sized.volume/(nx*L*ny*W**nz*H*design.DIMENSION**3)
        # logger.debug(f"Geometry relative density: {relativeDensity}")
        viz = dash_vtk.View(
            dash_vtk.GeometryRepresentation(
                [dash_vtk.Mesh(state=to_mesh_state(geom))],
                showCubeAxes=True,
                cubeAxesStyle={
                    "axisLabels": ["X", "Y", "Z"],
                    "axisTitlePixelOffset": 50,
                    "tickLabelPixelOffset": 25,
                    "gridLines": False,
                    "tickCounts": [3, 3, 3],
                },
            ),
        )
        logger.debug("Rendering complete")

        return html.Div(viz, style=dict(height="200px"))
    except Exception as ex:
        logger.debug(f"Unable to render geometry: {ex}")
        return []


# Define recursive custom evaluation function
def customEval(calcs, modeqn):

    # Evaluate the equation
    try:
        value = float(customeval(calcs, modeqn))
    except:
        # Assume this is an aspect ratio calculation where the
        # the equation variables haven't been updated to reference
        # the data object
        value = float(
            customeval(
                calcs,
                modeqn.replace("length", "data['length']")
                .replace("width", "data['width']")
                .replace("height", "data['height']"),
            )
        )

    return value


# @profile
def _createSubCardProps(unitcell, L, W, H, T, R, form, scustom, ssurrogate):
    """Create a score card of properties for a given design"""

    # Construct current options
    options = _options(scustom)

    # Construct custom surrogate equations
    logger.debug(f"Stored surrogate: {ssurrogate}")
    models = SURROGATE(ssurrogate, form, unitcell)
    logger.debug(f"Processed surrogate: {ssurrogate}")

    # Create table of properties
    try:
        header = [
            html.Thead(
                html.Tr([html.Th("QOI", style=dict(width="66%")), html.Th("Value")])
            )
        ]

        # Pull out the surrogate models for this unit cell
        # models = surrogate[form][unitcell]

        # Loop through each design parameter and calculate the response
        params = ["length", "width", "height", "thickness", "radius"]
        calcs = {"length": L, "width": W, "height": H, "thickness": T, "radius": R}
        X = np.array([[L, W, H, T, R]])
        rows = [
            html.Tr(
                [html.Td("Unitcell"), html.Td(unitcell, style={"text-align": "right"})]
            )
        ]

        # Loop through each QOI (including custom QOI) and use the surrogate
        # model to predict the property based on the given geometric properties.
        for param in columns:
            # Skip independent variables
            modifier = ""
            color = "green"
            info = ""
            if param in params:
                meanValue = float(calcs[param])
                uncertainty = None
                color = "green"
                info = ""
            elif "custom" in param:
                # Evaluate custom equation based on model predictions
                logger.debug(f"Custom equaluation: {param}, {models[param]['model']}")
                _, modeqn = models[param]["model"]
                meanValue = customEval(calcs, modeqn)
                uncertainty = None
                color = "red"
                info = (
                    "The custom QOI are calculated based on the "
                    "machine learning predictions for this "
                    "design. Currently, prediction uncertainties "
                    "aren't propagated; so, the underlying "
                    "definition should be checked to verify "
                    "the validity of the value."
                )
            elif "AR" in param:
                # Evaluate custom equation based on model predictions
                logger.debug(f"AS evaluation: {param}, {models[param]['model']}")
                _, modeqn = models[param]["model"]
                meanValue = customEval(calcs, modeqn)
                uncertainty = None
                color = "green"
                info = ""
            else:
                # Check cross validation scores
                model = models[param]["model"]
                scaling = models[param]["yscaler"]
                scores = models[param]["scores"]
                if scores.mean() > 0.9:
                    color = "green"
                else:
                    color = "red"
                    info = (
                        "The machine learning framework had issues "
                        "fitting this data set. The predicted values "
                        "likely have some error to them."
                    )

                # Compute model prediction
                value, std = model.predict(X, return_std=True)

                # Scale prediction to physical units
                meanValue = scaling.inverse_transform(np.array([value]))[0, 0]
                upper = scaling.inverse_transform(np.array([value + 1.96 * std]))[0, 0]
                lower = scaling.inverse_transform(np.array([value - 1.96 * std]))[0, 0]
                uncertainty = (upper - lower) / 2

                # Check values to highlight potentially incorrect numbers
                if meanValue < 0 and param[:2] != "nu":
                    color = "red"
                    info = (
                        "(Negative value doesn't make physical sense. "
                        "This is due to the machine learning approach. "
                        "The actual value is likely a very small "
                        "positive value.)"
                    )
                if uncertainty / np.abs(meanValue) > 0.2:
                    color = "red"
                    info = (
                        "The uncertainty in this machine learning "
                        "prediction is high."
                    )
                    modifier = f" ± {uncertainty:>.3n}"
                # logger.debug(f"Model: {param} {value} {std}")

            calcs[param] = meanValue

            # Create the tables rows: one for each QOI
            rows += [
                html.Tr(
                    [
                        html.Td(options[param]["name"], title=options[param]["info"]),
                        html.Td(
                            f"{meanValue:>.3n}" + modifier,
                            title=info,
                            style={"text-align": "right", "color": color},
                        ),
                    ]
                )
            ]

        stats = dbc.Table(header + rows, bordered=True, striped=True)

        return stats
    except Exception as ex:
        logger.debug(f"Unable to create score sheet: {ex}")
        return None


@app.callback(
    Output("download-custom", "data"),
    Input("save-custom", "n_clicks"),
    [
        State({"type": "custom-qoi-label", "qoi": ALL}, "value"),
        State({"type": "custom-qoi-eqn", "qoi": ALL}, "value"),
    ]
    + [
        State("qoi", "value"),
        State("normalization", "value"),
        State("extraFilters", "value"),
        State("forms", "value"),
        State({"type": "slider-qoiFilter", "qoi": ALL}, "value"),
    ],
    prevent_initial_call=True,
)
def downloadCustom(*args):
    # Get the triggering input
    ctx = dash.callback_context

    # Write the States
    return dict(
        content=json.dumps(ctx.states),
        type="application/json",
        filename="CustomQOI.json",
    )


@app.callback(
    Output({"type": "download-example", "example": MATCH}, "data"),
    Input({"type": "button-example", "example": MATCH}, "n_clicks"),
    prevent_initial_call=True,
)
def downloadExample(nclicks):
    # Get the triggering input
    ctx = dash.callback_context
    logger.debug(f"Download example file: {ctx.triggered}")

    # Determine the correct json file to download
    sprop, _ = ctx.triggered[0]["prop_id"].split("}.")
    prop = json.loads(sprop + "}")
    example = prop["example"]
    basename = Path(f"example-{example}.json")
    filename = Path(__file__).parent / Path("static/examples/") / basename
    logger.debug(f"File to download: {filename}")

    # Load the example data
    try:
        with open(filename, "r") as f:
            content = json.load(f)
    except Exception as ex:
        logger.error(f"Issue downloading example {basename}: {ex}")
        return dash.no_update

    return dict(
        content=json.dumps(content),
        type="application/json",
        filename=basename.as_posix(),
    )


@app.callback(
    [
        Output("load-custom-issue", "displayed"),
        Output("loadedConfig", "data"),
        Output("load-custom", "contents"),
    ],
    Input("load-custom", "contents"),
    [
        State({"type": "custom-qoi-label", "qoi": ALL}, "value"),
        State({"type": "custom-qoi-eqn", "qoi": ALL}, "value"),
    ]
    + [
        State("qoi", "value"),
        State("normalization", "value"),
        State("extraFilters", "value"),
        State("forms", "value"),
        State({"type": "slider-qoiFilter", "qoi": ALL}, "value"),
    ],
    prevent_initial_call=True,
)
def storeConfigFile(contents, *states):
    # Make sure there is a file
    logger.debug("Loading uploaded config file.")

    # Get the triggering input
    ctx = dash.callback_context

    # Convert base64 contents to dictionary
    try:
        _, contents = contents.split("base64")
        config = json.loads(base64.b64decode(contents))
    except Exception as ex:
        logger.debug(f"Unable to load uploaded file: {ex}")
        return True, dash.no_update, None

    # Check validity of config
    keys = set(config.keys())
    expected = set(ctx.states.keys())
    if keys != expected:
        logger.error(
            "Uploaded config file appears to be inconsistent "
            "with the application setup. The primary differences "
            f"are {expected-keys}"
        )
        return True, dash.no_update, None

    # Save the contents to file
    logger.debug(f"Loaded config: {config}")
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f)
    except:
        logger.debug(
            f"Unable to save config file {CONFIG_FILE} locally. There "
            "might be an issue with the file formatting. Make "
            "sure it is a proper json file."
        )
        return True, dash.no_update, None

    logger.debug(f"Saved config to {CONFIG_FILE}")

    return False, f"Saved config to {CONFIG_FILE}", None


@app.callback(
    [
        Output("qoi", "value"),
        Output("normalization", "value"),
        Output("extraFilters", "value"),
        Output("forms", "value"),
        Output({"type": "custom-qoi-label", "qoi": ALL}, "value"),
        Output({"type": "custom-qoi-eqn", "qoi": ALL}, "value"),
        Output("loadedFilters", "data"),
    ],
    Input("loadedConfig", "data"),
    prevent_initial_call=True,
)
def updateLoadedEquations(loaded):
    logger.debug("Custom QOI load request.")

    # Load the saved data
    with open(CONFIG_FILE, "r") as f:
        ref = json.load(f)
    logger.debug(f"Loaded config file {CONFIG_FILE}")

    # Pull out the labels and equations
    reflabels = [k for k in ref.keys() if "custom-qoi-label" in k]
    refeqns = [k for k in ref.keys() if "custom-qoi-eqn" in k]
    if len(reflabels) != NCUSTOM or len(refeqns) != NCUSTOM:
        logger.error(
            "The loaded custom definitions does not have "
            "the correct data. Data load failed"
        )
        return [dash.no_update] * (4 + NCUSTOM * 2 + 1)

    # Pull out options
    standard = [
        (
            ref[f"{id}.value"]
            if ref[f"{id}.value"] or ref[f"{id}.value"] == 0
            else dash.no_update
        )
        for id in ["qoi", "normalization", "extraFilters", "forms"]
    ]

    # Pull out custom equation labels and filter details
    labels = {}
    eqns = {}
    filters = {}
    for k, v in ref.items():
        if "custom-qoi" in k:
            # Custom equation definition
            definition = json.loads(k.split(".")[0])
            qoi = definition["qoi"]
            if "eqn" in k:
                eqns[qoi] = v
            else:
                labels[qoi] = v
        if "slider-qoiFilter" in k:
            # Slider range values
            definition = json.loads(k.split(".")[0])
            filters[definition["qoi"]] = v

    # Pull out the equation labels
    possible = [f"custom{i}" for i in range(1, NCUSTOM + 1)]
    labels = [labels[i] if i in labels.keys() else dash.no_update for i in possible]
    logger.debug(f"Loaded labels: {labels}")

    # Pull out custom equation definitions
    eqns = [eqns[i] if i in eqns.keys() else dash.no_update for i in possible]
    logger.debug(f"Loaded equations: {eqns}")

    return standard + [labels] + [eqns] + [loaded]


@app.callback(
    [Output(f"div-qoiFilter-{qoi}", "style") for qoi in columns],
    [Input("qoi", "value"), Input("extraFilters", "value")],
    [State(f"div-qoiFilter-{qoi}", "style") for qoi in columns],
)
def updateFilters(qois, extra, *current):
    # Determine desired state
    desired = {k: {"display": "none"} for k in columns}
    for k in qois + extra:
        desired[k] = {"display": "block"}

    # Determine the fields that need to change
    output = [
        dash.no_update if desire == ref else desire
        for desire, ref in zip(desired.values(), current)
    ]

    return output


@app.callback(
    [
        Output({"type": "slider-qoiFilter", "qoi": ALL}, "value"),
        Output({"type": "slider-qoiFilter-lower", "qoi": ALL}, "value"),
        Output({"type": "slider-qoiFilter-upper", "qoi": ALL}, "value"),
    ],
    [
        Input({"type": "slider-qoiFilter", "qoi": ALL}, "value"),
        Input({"type": "slider-qoiFilter-lower", "qoi": ALL}, "value"),
        Input({"type": "slider-qoiFilter-upper", "qoi": ALL}, "value"),
        Input({"type": "slider-qoiFilter", "qoi": ALL}, "min"),
        Input({"type": "slider-qoiFilter", "qoi": ALL}, "max"),
        Input("qoi", "value"),
        Input("extraFilters", "value"),
        Input("loadedFilters", "data"),
    ],
    prevent_initial_call=True,
)
def updateFiltersFromInputs(
    ivalues, ilowers, iuppers, imins, imaxs, qois, extraFilters, loaded
):
    # @TODO This is a fragile implementation. Right now, it assumes the
    # input order is always consistent. A more robust way to do this
    # would be using linked naming.

    # Get the triggering input
    ctx = dash.callback_context
    logger.debug(f"Updating triggered filter: {ctx.triggered}")

    # Set all outputs to no update by default
    ovalues = [dash.no_update] * len(ivalues)
    olowers = [dash.no_update] * len(ilowers)
    ouppers = [dash.no_update] * len(iuppers)

    # Check to see if a configuration was loaded. If so, update all
    if [t for t in ctx.triggered if "loadedFilters" in t["prop_id"]]:
        logger.debug(loaded)
        # Load the config data
        with open(CONFIG_FILE, "r") as f:
            ref = json.load(f)
        logger.debug(f"Loaded config file {CONFIG_FILE}")

        # Pull out options
        standard = [
            (
                ref[f"{id}.value"]
                if ref[f"{id}.value"] or ref[f"{id}.value"] == 0
                else dash.no_update
            )
            for id in ["qoi", "normalization", "extraFilters", "forms"]
        ]
        # Pull out custom equation labels and filter details
        filters = {}
        for k, v in ref.items():
            if "slider-qoiFilter" in k:
                # Slider range values
                definition = json.loads(k.split(".")[0])
                filters[definition["qoi"]] = v

        # Only update filters specified in the qoi and extraFilters
        ovalues = [
            (
                v
                if (
                    k in standard[0]
                    or k in ([] if standard[2] == dash.no_update else standard[2])
                )
                else dash.no_update
            )
            for k, v in filters.items()
        ]
        olowers = [dash.no_update if v == dash.no_update else v[0] for v in ovalues]
        ouppers = [dash.no_update if v == dash.no_update else v[1] for v in ovalues]
        # return ovalues, olowers, ouppers
    elif "slider-qoiFilter" in ctx.triggered[0]["prop_id"]:
        # The filters were manually updated. Only update the modified
        # filter

        # Update values based on triggering entity
        triggers = [
            (json.loads(t["prop_id"].split(".")[0]), t["prop_id"].split(".")[1])
            for t in ctx.triggered
            if "slider-qoiFilter" in t["prop_id"]
        ]
        tinputs = [
            t for t in triggers if "lower" in t[0]["type"] or "upper" in t[0]["type"]
        ]
        cinputs = [t for t in triggers if "min" in t[1] or "max" in t[1]]

        # Pull out the target qoi
        targetqoi = triggers[0][0]["qoi"]
        logger.debug(f"Triggering filter QOI: {targetqoi}")

        # Created an indexed list of what needs to be updated
        toupdate = [
            True if targetqoi == qoi["id"]["qoi"] else False
            for i, qoi in enumerate(ctx.inputs_list[0])
        ]

        ovalues = []
        olowers = []
        ouppers = []
        for i, check in enumerate(toupdate):
            # Pull that the state based on the index
            if check:
                ivalue = ivalues[i]
                ilower = ilowers[i]
                iupper = iuppers[i]
                imin = imins[i]
                imax = imaxs[i]
            else:
                # No update required. Specify as such and continue
                ovalues.append(dash.no_update)
                olowers.append(dash.no_update)
                ouppers.append(dash.no_update)
                continue

            # Process the input based on the intput type
            try:
                olower = dash.no_update
                oupper = dash.no_update
                ovalue = dash.no_update
                if tinputs:
                    # An input was mofidied, so update the sliders accordingly
                    logger.debug("Filter input text modified. Updating sliders.")
                    lower = float(ilower)
                    upper = float(iupper)
                    if lower < float(imin):
                        lower = float(imin)
                        olower = lower
                    if upper > float(imax):
                        upper = float(imax)
                        oupper = upper
                    if lower > upper:
                        lower = upper
                        olower = oupper = lower
                    ovalue = [lower, upper]
                elif cinputs:
                    # One of the custom QOI equations definitions changed. Update
                    # the bounds and set the slider to the full extent
                    logger.debug(
                        "Custom QOI modified. Updating sliders and input boxes."
                    )
                    ovalue = [float(imin), float(imax)]
                    olower = float(imin)
                    oupper = float(imax)

                else:
                    # Otherwise, a slider changed and the corresponding inputs
                    # need to be updated
                    logger.debug(f"Sliders changed. Updating text box values: {ivalue}")
                    olower = float(ivalue[0])
                    oupper = float(ivalue[1])
            except Exception as ex:
                logger.debug(
                    "Slider value or inputs triggered but couldn't "
                    f"parse the update: {ex}"
                )
                ovalue = ivalue
                olower = ilower
                oupper = iupper

            ovalues.append(ovalue)
            olowers.append(olower)
            ouppers.append(oupper)

    # Make sure all non visible filters are set to their bounds. This
    # prevents hidden filters from reducing the plotted data without the
    # user being aware of it.
    for i, qoidict in enumerate(ctx.inputs_list[0]):
        qoi = qoidict["id"]["qoi"]
        if not (qoi in qois or qoi in extraFilters):
            # QOI is hidden and should be checked
            if np.isclose(ilowers[i], imins[i]) and np.isclose(iuppers[i], imaxs[i]):
                # Bounds are consistent with the min and max values.
                pass
            else:
                logger.debug(
                    f"The QOI filter {qoi} is not visable, "
                    f"but the slider range {ivalues[i]} "
                    f"hasn't been reset to [{imins[i]}, "
                    f"{imaxs[i]}]. Doing so now."
                )
                ovalues[i] = [imins[i], imaxs[i]]
                olowers[i] = imins[i]
                ouppers[i] = imaxs[i]

    logger.debug(f"Filters updated: {ovalues}, {olowers}, {ouppers}")
    return ovalues, olowers, ouppers


# @TODO improve callback efficiency by only updating the needed outputs
@app.callback(
    [
        Output("qoi", "options"),
        Output("normalization", "options"),
        Output("customOptions", "data"),
        Output("extraFilters", "options"),
    ]
    + [
        Output({"type": "slider-qoiFilter-label", "qoi": f"custom{ind}"}, "children")
        for ind in range(1, NCUSTOM + 1)
    ],
    [
        Input("qoi", "value"),
        Input({"type": "custom-qoi-label", "qoi": ALL}, "value"),
    ],
    State("customOptions", "data"),
)
def updateOptions(values, labels, scustom):
    # Get current options, falling back to defaults if custom options haven't
    # been initialized yet.
    if scustom:
        options = _options(scustom)
    else:
        logger.info(
            "The custom user options haven't been stored yet. Falling "
            "back to defaults."
        )
        options = _options(_DEFAULT_CUSTOM)

    # Default to no slider updates and only update those that change
    sliderLabels = [dash.no_update for i in range(len(labels))]
    for i, label in enumerate(labels):
        if label:
            logger.debug(f"Labeling 'custom{i+1}' as {label}")
            options[f"custom{i+1}"]["name"] = label
            sliderLabels[i] = label

    # Serialize updated custom options
    scustom = json.dumps(
        {
            k.split("custom")[1]: {subk: v[subk] for subk in ["name", "info"]}
            for k, v in options.items()
            if "custom" in k
        }
    )

    logger.debug(f"Slider labels: {sliderLabels}")
    return (
        OPTIONS(options),
        OPTIONS_NORMALIZE(options),
        scustom,
        [o for o in OPTIONS(options) if o["value"] not in values],
        *sliderLabels,
    )


@app.callback(
    [
        Output({"type": "slider-qoiFilter", "qoi": f"custom{ind}"}, "min")
        for ind in range(1, NCUSTOM + 1)
    ]
    + [
        Output({"type": "slider-qoiFilter", "qoi": f"custom{ind}"}, "max")
        for ind in range(1, NCUSTOM + 1)
    ]
    + [
        Output({"type": "slider-qoiFilter", "qoi": f"custom{ind}"}, "step")
        for ind in range(1, NCUSTOM + 1)
    ]
    + [
        Output({"type": "custom-qoi-eqn", "qoi": ALL}, "valid"),
        Output({"type": "custom-qoi-eqn", "qoi": ALL}, "invalid"),
        Output("customSurrogates", "data"),
        Output("customBounds", "data"),
        Output("customData", "data"),
    ],
    Input({"type": "custom-qoi-eqn", "qoi": ALL}, "value"),
    [
        State("customSurrogates", "data"),
        State("customBounds", "data"),
        State("customData", "data"),
    ],
    prevent_initial_call=True,
)
def updateCustomQOI(eqns, ssurrogate, sbounds, sdata):
    # @TODO: The filter bounds for nested custom variables doesn't get updated properly.

    # # Construct current custom surrogate models
    # surrogate = SURROGATE(ssurrogate)

    # Create a dictionary to keep track of the updated custom surrogate models
    try:
        newSurrogates = json.loads(ssurrogate)
    except:
        # If the input is empty, set as default parameters
        newSurrogates = [_SURROGATE_DEFAULT for _ in range(NCUSTOM)]
    newSurrogates = {i + 1: v for i, v in enumerate(newSurrogates)}

    # Construct current custom bounds
    bounds = BOUNDS(sbounds)

    # Create a dictionary to keep track of the updated custom bounds
    try:
        newBounds = json.loads(sbounds)
    except:
        # If the input is empty, set as default parameters
        newBounds = [_BOUNDS_DEFAULT for _ in range(NCUSTOM)]
    newBounds = {i + 1: v for i, v in enumerate(newBounds)}

    # Construct full data structure with custom definitions
    data = DATA(sdata)

    # Set default no update outputs
    vmins = [dash.no_update] * NCUSTOM
    vmaxs = [dash.no_update] * NCUSTOM
    steps = [dash.no_update] * NCUSTOM

    # Make sure the valid/invalid markers are cleared if the equation
    # reference is empty. Otherwise, default to a no update state
    valids = [dash.no_update if eqn else False for eqn in eqns]
    invalids = [dash.no_update if eqn else False for eqn in eqns]

    # @TODO This is fragile and breaks if there are more
    # than 9 custom variables. Would need to add regular expression
    # into the processing.
    # assert NCUSTOM < 10, "Doesn't currently support more than 9 custom variables."

    # Pull out the custom equation index number
    ctx = dash.callback_context

    # Determine the triggering input
    logger.debug(f"Custom QOI equation updated: {ctx.triggered}")
    props = [json.loads(trig["prop_id"].split(".value")[0]) for trig in ctx.triggered]
    qois = [prop["qoi"] for prop in props]
    qoi = qois[0]
    inds = [int("".join(qoi.split("custom")[1:])) for qoi in qois]

    # Pull out all of the variables used in each custom equation
    if inds:
        modeqns = []
        for eqn in eqns:
            if eqn:
                # Tokenize the input equation and modify it with referenes
                # to the dataframe object
                g = tokenize(BytesIO(eqn.encode("utf-8")).readline)
                result = []
                variables = []
                for toknum, tokval, _, _, _ in g:
                    # If the current token is a NAME and in the possible variable list, then
                    # the updated the representation to be wrapped in the substition code
                    if toknum == NAME and tokval in columns:
                        variables.append(tokval)
                        result.extend([(STRING, f"data['{tokval}']")])
                    else:
                        # Otherwise, use the current representation
                        result.append((toknum, tokval))
                modeqn = untokenize(result).decode("utf-8")
                logger.debug(f"Modified equation: {variables}, {modeqn}")
                modeqns.append((variables, modeqn))
            else:
                # Blank equation
                modeqns.append(([], eqn))

    # If a custom variable is referenced within a custom variable, there
    # is the posibility that one custom variable will get calculated
    # before the other is updated, resulting in an incorrect
    # calculation. To combat this, we run the calculations twice if a
    # custom variable is found in an equation definition. Note, this
    # doesn't look for or fix circular definitions.
    newinds = []
    for j in range(2):
        # If recalculations are required, update the required equations
        if newinds:
            logger.debug(
                f"QOI {qoi} is referenced by other custom equations: "
                f"{newinds}. Updating them as well."
            )
            inds = list(set(newinds))

        # Loop through the requested equations that need to be updated
        for ind in inds:
            variables, modeqn = modeqns[ind - 1]
            eqn = eqns[ind - 1]
            if modeqn:
                try:
                    # Check for recursive definitions
                    if f"custom{ind}" in variables:
                        logger.error(
                            f"Custom variable 'custom{ind}' is referenced "
                            "within it's own equation, which is not "
                            f"allowed: {eqn}"
                        )
                        invalid = True
                        valid = False
                        raise RuntimeError("Circular reference")

                    # Check for other equations that reference this custom
                    # variable
                    logger.debug(f"Equations: {eqns}, {ind}")
                    newinds += [
                        k + 1
                        for k, (subvariables, subeqn) in enumerate(modeqns)
                        if subeqn and k != ind - 1 and (f"custom{ind}" in subvariables)
                    ]

                    # Modify the equation string to appropriately reference the
                    # precomputed data.
                    newSurrogates[ind] = (variables, modeqn)

                    logger.debug(f"Custom QOI {ind} eqn: {modeqn}")

                    lowerBound = 1e10
                    upperBound = -1e10
                    # Update the QOI for each form and unitcell
                    for k in data.keys():
                        for subk in data[k].keys():
                            evaluated = customeval(data[k][subk], modeqn)
                            # if np.any(np.isnan(evaluated)):
                            #     logger.debug(
                            #         f"NaNs in custom QOI {ind}: {evaluated}, {data[k][subk]}"
                            #     )
                            #     error = f"Custom QOI {ind} equation {modeqn} resulted in NaN values."
                            #     logger.error(error)
                            #     raise RuntimeError(error)
                            data[k][subk][f"custom{ind}"] = evaluated

                            # Stack up the column based bounds across all datasets
                            submin = data[k][subk][f"custom{ind}"].min()
                            submax = data[k][subk][f"custom{ind}"].max()
                            lowerBound = np.minimum(submin, lowerBound)
                            upperBound = np.maximum(submax, upperBound)

                            valid = True
                            invalid = False
                    newBounds[ind] = [lowerBound, upperBound]
                except:
                    logger.debug(
                        f"Calculating Custom QOI {ind} failed. Reverting to detaults."
                    )
                    for k in data.keys():
                        for subk in data[k].keys():
                            data[k][subk][f"custom{ind}"] = 1

                    newBounds[ind] = _BOUNDS_DEFAULT
                    valid = False
                    invalid = True

                # vmin, vmax = BOUNDS[f"custom{ind}"]
                vmin, vmax = newBounds[ind]
                if np.isclose(vmin, vmax):
                    # If the bounds are the same, offset them slightly to prevent a
                    # divide by zero issue
                    vmin *= 0.9
                    vmax *= 1.1
                scaling = 10 ** np.round(np.log10(vmax - vmin) - 2)
                vmins[ind - 1] = np.floor(vmin / scaling) * scaling
                vmaxs[ind - 1] = np.ceil(vmax / scaling) * scaling
                steps[ind - 1] = (vmax - vmin) / 100
                valids[ind - 1] = valid
                invalids[ind - 1] = invalid

    logger.debug(
        f"Custom QOI {ind} updated: {vmins[ind-1]}, " f"{vmaxs[ind-1]}, {steps[ind-1]}."
    )

    # Parse out the new custom data values and package them for storage
    newData = {
        form: {
            unitcell: df[[f"custom{i+1}" for i in range(NCUSTOM)]].to_dict()
            for unitcell, df in data[form].items()
        }
        for form in data.keys()
    }

    return (
        vmins
        + vmaxs
        + steps
        + [valids]
        + [invalids]
        + [json.dumps([v for v in newSurrogates.values()])]
        + [json.dumps([v for v in newBounds.values()])]
        + [json.dumps(newData)]
    )


@app.callback(
    Output("graphAshbyPlotsTooltip", "show"),
    Output("graphAshbyPlotsTooltip", "bbox"),
    Output("graphAshbyPlotsTooltip", "children"),
    Output("graphAshbyPlotsTooltip", "background_color"),
    Output("graphAshbyPlotsTooltip", "border_color"),
    Input("graphAshbyPlots", "hoverData"),
    [State("curves", "data")],
    prevent_initial_call=True,
)
def updateToolTips(hoverData, scurves):
    # Hide hover data if no points hovered over
    if hoverData is None:
        return False, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Define format string for unitcell definition
    FORMAT = (
        "{unitcell}(L={L:.1f}, W={W:.1f}, H={H:.1f}, "
        "T={T:.2f}, R={R:.2f}, form={form})"
    )

    # Pull out the curves data
    try:
        CURVES = json.loads(scurves)
    except:
        CURVES = []

    # Get first point
    # @TODO show multiple tooltips if relevant. This will require
    # switching hovermode to "x" or "y". This will trigger more items
    # then desired, so the the corresponding "x" or "y" values will need
    # to be checked to verify overlapping points.
    # logger.debug(f"Number of hover points: {hoverData}")
    count = 0
    for point in hoverData["points"]:
        bbox = point["bbox"]

        curveNumber = point["curveNumber"]
        pointNumber = point["pointNumber"]
        x, y = [point[v] for v in "xy"]
        index = point["customdata"]

        # Get unitcell details, noting that the CURVES entries are in the
        # form "<unitcell> (<form>)"
        curve = CURVES[curveNumber]
        split = curve.split(" (")
        unitcell = split[0]
        form = split[1][:-1].replace(" ", "")
        parameters = _DATA[form][unitcell][
            ["length", "width", "height", "thickness", "radius"]
        ].iloc[index]
        definition = {param: v for param, v in zip("LWHTR", parameters)}
        definition["unitcell"] = unitcell
        definition["form"] = form

        image = IMAGES[form][unitcell]()[index]

        # Pull out curve color. Not, because there is a curvne number spans
        # all of the subplots, we're just interested in the first
        # occurrence.
        subCurveNumber = [i for i in range(curveNumber + 1) if CURVES[i] == curve][0]
        colorcycler = cycle(DEFAULT_PLOTLY_COLORS)
        color = [next(colorcycler) for _ in range(subCurveNumber + 1)][-1]

        # Create output child
        if count == 0:
            # For primary selection, highlight the unit cell and what it
            # looks like
            xref = x
            yref = y
            L, W, H, T, R = parameters
            style = {"color": "black"}
            children = [
                html.H4(f"{curve}", style=style),
                html.Img(src=image),
                html.P(
                    [
                        f"x: {x:.3e}",
                        html.Br(),
                        f"y: {y:.3e}",
                        html.Br(),
                        f"Length: {L:.1f}",
                        html.Br(),
                        f"Width: {W:.1f}",
                        html.Br(),
                        f"Height: {H:.1f}",
                        html.Br(),
                        f"Thickness: {T:.1f}",
                        html.Br(),
                        f"Smoothing radius: {R:.1f}",
                    ],
                    style=style,
                ),
                # html.P(FORMAT.format(**definition)),
            ]
            count += 1
        else:
            # Check if the point has similar (x, y) definition to the
            # primary points
            if np.isclose(x, xref) and np.isclose(y, yref):
                # Add in similar definitions if present
                if count == 1:
                    children += [html.H4("Similar")]
                children += [html.Br(), html.P(FORMAT.format(**definition))]

                count += 1

        # For now, short circuit the for loop to cut down on processing time.
        break

    return True, bbox, children, color, color


# Take selection data and update the figure, hiding all other points
# and labeling them in their selection order
def _selectionUpdate(fig, selected, curves):
    # Parse out point data for legend group
    parsed = {group: [] for group in curves}
    for i, s in enumerate(selected["points"]):
        try:
            parsed[curves[s["curveNumber"]]].append((s["pointNumber"], i))
        except:
            parsed[curves[s["curveNumber"]]] = [(s["pointNumber"], i)]

    # Update trace visibility
    for group, points in parsed.items():
        logger.debug(f"{group}: {points}")

        # Create point labeling based on selection order
        if points:
            # For each point, create an indexing label
            ps = [p[0] for p in points]
            text = [""] * (max(ps) + 1)
            for p in points:
                text[p[0]] = f"{p[1]+1}"
        else:
            # If there are no points selected for this curve,
            # output blank text and selection arrays
            text = None
            ps = []

        # Update traces based on their curve number
        fig.update_traces(
            selector=dict(legendgroup=group),
            selectedpoints=ps,
            text=text,
            mode="markers+text",
            textposition="bottom center",
        )

    return fig


@app.callback(
    [
        Output("graphAshbyPlots", "figure"),
        Output("doubleClick", "data"),
        Output("selectedData", "data"),
        Output("curves", "data"),
    ],
    [
        Input("qoi", "value"),
        Input("forms", "value"),
        Input("graphAshbyPlots", "selectedData"),
        Input("graphAshbyPlots", "clickData"),
        #  Input('wrapperGraphAshbyPlots', 'n_clicks'),
        Input("resetSelection", "n_clicks"),
        Input("normalization", "value"),
        Input("extraFilters", "value"),
        Input({"type": "slider-qoiFilter", "qoi": ALL}, "value"),
        Input({"type": "custom-qoi-label", "qoi": ALL}, "value"),
    ],
    [
        State("graphAshbyPlots", "figure"),
        State("selectedData", "data"),
        State("customOptions", "data"),
        State("curves", "data"),
        State("customData", "data"),
    ],
)
# @profile
def updateGraphAshbyPlots(
    values,
    forms,
    selectedData,
    clickData,
    nclicks,
    normalization,
    extraFilters,
    filter,
    customLabel,
    fig,
    selectedDataStored,
    scustom,
    scurves,
    sdata,
):
    ctx = dash.callback_context
    logger.debug(f"Updating Ashby plots: {ctx.triggered}")

    # Parse out inputs
    # fig, selectedDataStored, doubleClick = args[-3:]
    # filters = args[:len(columns)]

    # Get the triggering input
    try:
        trigger = [t["prop_id"] for t in ctx.triggered]
    except:
        trigger = None
    if trigger is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # If the update corresponds to a custom label that is not
    # currently being plotted, break out early
    if len(trigger) == 1 and "custom-qoi-label" in trigger[0]:
        logger.debug("Custom QOI label update triggered graph update.")
        customQOI = json.loads(trigger[0].split(".")[0])["qoi"]
        if customQOI not in values:
            logger.debug(
                f"Custom QOI '{customQOI}' not in the list of "
                f"of QOI to be plotted: {values}. Skipping "
                "plot update."
            )
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # # Update the filter options based on the QOI values
    # options = [o for o in OPTIONS if o["value"] not in values]

    # Pull out the current filter bounds
    bounds = {state["id"]["qoi"]: state["value"] for state in ctx.inputs_list[-2]}

    # Check for clearing click event (i.e., click event not on a data point)
    if any(["n_clicks" in t for t in trigger]) and len(trigger) == 1:
        if selectedDataStored:
            fig = go.Figure(fig)
            fig.update_traces(selectedpoints=None, text=None)
            return fig, dash.no_update, [], dash.no_update
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        # return PreventUpdate
        # pass

    # Check that there are enough QOI to plot
    N = len(values)
    if N < 2:
        del fig
        return BLANK_FIGURE, dash.no_update, dash.no_update, json.dumps([])

    # # If a figure already exists and data was clicked, highlight the
    # # point and exist early
    selected = selectedDataStored or {"points": []}
    # # # logger.debug(f"Click: {oldClickData}, {clickData}")
    if any(["clickData" in t for t in trigger]):
        # logger.debug(f"Registered click data: {clickData}")
        selected["points"] += clickData["points"]
    elif any(["selectedData" in t for t in trigger]):
        # selected = selectedData
        selected["points"] += selectedData["points"]
    logger.debug(f"Selected points: {selected}")

    # Extract the current state of the figure curves
    try:
        curves = json.loads(scurves)
    except:
        curves = []

    if (
        any(["selected" in t or "click" in t for t in trigger])
        and fig
        and selected["points"]
    ):
        # selected = selectedData['points']
        # Convert figure from a dictionary to a figure object
        fig = go.Figure(fig)

        # Add in text annotations linking each pointed to its score sheet
        fig = _selectionUpdate(fig, selected, curves)

        return fig, dash.no_update, selected, dash.no_update

    # Create a grid of subplots
    fig = subplots.make_subplots(
        rows=N - 1, cols=N - 1, shared_xaxes=True, shared_yaxes=True
    )
    # Loop through and add in traces
    showlegend = True

    # Reset the curves list
    curves = []

    # Get current options, falling back to defaults if custom options haven't
    # been initialized yet.
    if scustom:
        options = _options(scustom)
    else:
        options = _options(_DEFAULT_CUSTOM)

    # Get current data
    data = DATA(sdata)

    # # Reclassify the graph and corrugation forms
    # formKeys = {"graph": [], "walledtpms": []}
    # for k in forms:
    #     if "graph" in k:
    #         formKeys['graph'].extend([subk for subk in data['graph'].keys() if "honeycomb" not in subk.lower()])
    #         logger.debug(f"Form {k}: {formKeys['graph']}")
    #     elif "corrugation" in k:
    #         formKeys['graph'].extend([subk for subk in data['graph'].keys() if "honeycomb" in subk.lower()])
    #         logger.debug(f"Form {k}: {formKeys['graph']}")
    #     else:
    #         formKeys['walledtpms'] = data[k].keys()
    # forms = list(set("graph" if "corrugation" in k else k for k in forms))
    # logger.debug(f"Forms: {forms}")

    for r in range(1, N):
        logger.debug(f"Updating row {r}")
        param1 = values[r]
        label1 = options[param1]["name"]
        if normalization != 0:
            label1 = "Normalized " + label1
        for c in range(r):
            logger.debug(f"Updating column {c}")
            param2 = values[c]
            label2 = options[param2]["name"]
            if normalization != 0:
                label2 = "Normalized " + label2
            counter = 0
            colorcycler = cycle(DEFAULT_PLOTLY_COLORS)
            markercycler = cycle(MARKER_SYMBOLS)
            for k in forms:
                for subk in data[k].keys():
                    # if "honeycomb" in subk:
                    #     f = f.replace("graph", "corrugation")
                    # elif "tpms" in k:
                    #     f = "walled tpms"
                    # else:
                    #     f = k

                    name = f"{subk} ({k})"
                    curves.append(name)
                    color = next(colorcycler)
                    marker = next(markercycler)

                    # Filter the data according to the specified filter
                    # ranges
                    inds = np.logical_and.reduce(
                        [
                            bounds[p][0] <= data[k][subk][p]
                            for p in values + extraFilters
                        ]
                        + [
                            data[k][subk][p] <= bounds[p][1]
                            for p in values + extraFilters
                        ]
                    )
                    # xmin, xmax = bounds[param2]
                    # ymin, ymax = bounds[param1]
                    x = data[k][subk][param2].copy()
                    y = data[k][subk][param1].copy()

                    # If specified, normalized the qoi
                    if normalization and normalization != 0:
                        norm = data[k][subk][normalization]
                        x /= norm
                        y /= norm

                    # inds = np.logical_and.reduce([xmin <= x, x <= xmax,
                    #                               ymin <= y, y <= ymax])

                    # logger.debug(inds)
                    # This sequence was resulting in a significant memory leak.
                    # Adding in the pandas conversion to numpy seemed to stem the
                    # leak, although it still seems to be mildly present.
                    # https://github.com/pola-rs/polars/issues/18074
                    trace = go.Scattergl(
                        x=x[inds].to_numpy(),
                        y=y[inds].to_numpy(),
                        mode="markers",
                        marker=dict(color=color, symbol=marker),
                        name=name,
                        legendgroup=name,
                        showlegend=showlegend,
                        customdata=data[k][subk].index[inds],
                    )
                    fig.add_trace(trace, row=r, col=c + 1)
                    del x, y, trace
                    counter += 1
            showlegend = False

            # If the plot is in the first, label the x axis
            if r == N - 1:
                fig.update_xaxes(title=label2, row=r, col=c + 1)

            # If the plot is in the first column, label the y axis
            if c == 0:
                # I like to have the y axis text horizontal. This can be
                # a little challenging to get positioned correctly, so
                # I manually add in the annotation

                # Select the correct y domain based on the graphs location
                # in the subplot grid.
                if r == 1:
                    yref = "y domain"
                else:
                    yref = f"y{(r-1)*(N-1)+1} domain"
                # logger.debug(f"Y label: {yref} {label1}")

                # Add the annotation
                fig.add_annotation(
                    xref="x domain",
                    yref=yref,
                    x=0,
                    y=0.9,
                    axref="pixel",
                    ayref="pixel",
                    ax=-80,
                    ay=0,
                    arrowcolor="rgba(255, 255, 255, 0)",
                    text="<br>".join(textwrap.wrap(label1, 15)),
                    xanchor="right",
                    align="left",
                    yanchor="top",
                    valign="top",
                )
                # label = "<br>".join(label1.split(" "))
                # fig.update_yaxes(tickprefix=label+"  ", row=r, col=c+1)

    # Explicitly clear data to minimize memory leaks
    data.clear()

    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0),
        height=1000,
        hovermode="closest",
        margin=dict(l=200),
        clickmode="event",
    )
    #   clickmode="event+select")
    # fig = BLANK_FIGURE

    fig.update_traces(
        unselected={"marker": {"opacity": 0.1}},
        #  'color': 'rgb(100, 100, 100)' }},
        selected={"marker": {"size": 10}},
    )

    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
    )

    # If there are selected points, go through and rehighlight them
    if selected["points"]:
        fig = _selectionUpdate(fig, selected, curves)

    return fig, dash.no_update, selected, json.dumps(curves)


@app.callback(
    Output(f"selectedComparison", "children"),
    Input("selectedData", "data"),
    [
        State("customOptions", "data"),
        State("curves", "data"),
        State("customData", "data"),
    ],
)
def updateSelectedComparison(selected, scustom, scurves, sdata):
    # Check to see if there are any selected points. If not, exit early.
    if not selected:
        return []

    # Pull out the current options
    options = _options(scustom)

    # Pull out the current curved data
    try:
        CURVES = json.loads(scurves)
    except:
        CURVES = []

    # Get current data
    data = DATA(sdata)

    # Loop through each selected point, maxing out at 3 points, and
    # aggregate the QOI data.
    output = []
    for i, point in zip(range(3), selected["points"]):
        # Determine the relevant unitcell definition based on the
        # selected point
        point = selected["points"][i]
        curveNumber = point["curveNumber"]
        index = point["customdata"]
        curve = CURVES[curveNumber]
        split = curve.split(" (")
        unitcell = split[0]
        form = split[1][:-1]
        # form = "graph" if form in ["truss", "corrugation"] else form
        parameters = data[form][unitcell][
            ["length", "width", "height", "thickness", "radius"]
        ].iloc[index]
        L, W, H, T, R = parameters

        # Read in the image thumbnail for the unitcell
        image = IMAGES[form][unitcell]()[index]

        # Pull out the relevant performance metrics and tabulate them
        df = data[form][unitcell]
        pairs = zip(
            ["length", "width", "height", "thickness", "radius"], [L, W, H, T, R]
        )

        # Check that a datapoint exists in the database
        conditions = np.all([np.isclose(df[k], v) for k, v in pairs], axis=0)

        if np.any(conditions):
            # If an exact match is found, pull out relevant performance
            # metrics
            logger.debug(
                "Found design in the database. Creating a table "
                "of performance metrics."
            )
            perf = df.loc[conditions].iloc[0]
            output.append((image, curve, perf))
            # rows += [html.Tr([html.Td(_OPTIONS[col]["name"], title=_OPTIONS[col]["info"]),
            #                  html.Td(f"{perf.loc[col]:>.3n}", style={"text-align":"right"})])
            #                                             for col in perf.index]
        else:
            # Otherwise, pull out relevant geometric properties
            # col = "relativeDensity"
            # rows += [html.Tr([html.Td(_OPTIONS[col]["name"], title=_OPTIONS[col]["info"]),
            # html.Td(f"{relativeDensity:>.3n}",
            # style={"text-align":"right"})])]
            pass

    # Explicitly clear data to minimize memory leaks.
    data.clear()
    del data

    # If output data exists, form the data into a unified table for comparison
    # @TODO In the future, it would be nice to convert the comparison over
    # to bar chart representation to give a better sense of the relative
    # scaling differences
    if output:
        # Pull out the first set of data for reference
        ref = output[0][2]

        columns = ["QOI"] + [f"Selected {i+1}" for i in range(len(output))]

        data = [[("", "")] + [html.Img(src=d[0]) for d in output]]

        data += [
            [("Unitcell", "Form of the underlying periodicity.")]
            + [d[1] for d in output]
        ]

        data += [
            [(options[col]["name"], options[col]["info"])] + [d[2][col] for d in output]
            for col in ref.index
        ]

        # df = pd.DataFrame(data=data, columns=columns)

        # logger.debug(df)

        # data += [merge({"qoi": _OPTIONS[col]["name"]},
        #                {f"selected{i+1}": d[2][col]
        #                     for i, d in enumerate(output)})
        #                             for col in ref.index]

        def toTd(row):
            out = [html.Td(row[0][0], title=row[0][1])]
            try:
                out += [html.Td(f"{x::>.3n}", title=row[0][1]) for x in row[1:]]
            except:
                out += [html.Td(x, title=row[0][1]) for x in row[1:]]
            return out

        header = [html.Thead([html.Td(html.H4(col)) for col in columns])]
        rows = [html.Tbody([html.Tr(toTd(row)) for row in data])]

        # return dash_table.DataTable(data=data, columns=columns,
        #                             markdown_options={"html": True},)
        # return dbc.Table.from_dataframe(df)

        out = [
            html.H3(children="Selection detailed comparison"),
            html.Br(),
            dbc.Table(header + rows, striped=True, hover=True),
        ]

        data.clear()
        del options, data
        return out
    else:
        return []


@app.callback(
    [
        Output({"type": "card-T", "index": MATCH}, "className"),
        Output({"type": "card-T-issue", "index": MATCH}, "displayed"),
    ],
    Input({"type": "card-T", "index": MATCH}, "value"),
    Input({"type": "card-T", "index": MATCH}, "marks"),
    [
        State("customData", "data"),
        State({"type": "card-unitcell", "index": MATCH}, "value"),
    ],
)
def updateAlertThicknessColor(thickness, marksT, sdata, unitcellForm):
    try:
        # Parse the unitcell type and form
        if unitcellForm:
            split1 = unitcellForm.split(" (")
        else:
            return None, False
        unitcell = split1[0]
        form = split1[1][:-1].replace(" ", "").lower()
        try:
            # Get the current data
            if marksT is None:
                data = DATA(sdata)

                # Determine the appropriate slider markings for the given
                # unitcell.
                df = data[form][unitcell]
                marksT = {v: f"{v}" for v in df["thickness"].dropna().unique()}
                # marksR = {v: f"{v}" for v in df['radius'].unique()}

            # If the current thickness value is out of bounds from the
            # parametric study, shift it to be within bounds
            ts = np.array([float(t) for t in marksT.keys()])

            # If the thickness value is outside of the parametric study
            # bounds, indicate that there is a potential issue.
            if float(thickness) < ts.min() or float(thickness) > ts.max():
                return "danger", True

        except Exception as ex:
            logger.error(
                "Unable to determine precomputed thickness "
                f"and fillet radius for {unitcell} ({form}): "
                f"{ex}"
            )
    except Exception as ex:
        logger.error(
            "Unable to update tick markings based on " f"unitcell {unitcellForm}: {ex}"
        )

    return None, False


@app.callback(
    [
        Output({"type": "card-T", "index": MATCH}, "marks"),
        Output({"type": "card-T", "index": MATCH}, "value"),
    ],
    Input({"type": "card-unitcell", "index": MATCH}, "value"),
    [State("customData", "data"), State({"type": "card-T", "index": MATCH}, "value")],
    prevent_initial_call=True,
)
def updateCardMarks(unitcellForm, sdata, thickness):
    value = dash.no_update
    try:
        # Parse the unitcell type and form
        split1 = unitcellForm.split(" (")
        unitcell = split1[0]
        form = split1[1][:-1].replace(" ", "").lower()

        try:
            # Get the current data
            data = DATA(sdata)

            # Determine the appropriate slider markings for the given
            # unitcell.
            df = data[form][unitcell]
            marksT = {v: f"{v}" for v in df["thickness"].dropna().unique()}
            # marksR = {v: f"{v}" for v in df['radius'].unique()}

            # If the current thickness value is out of bounds from the
            # parametric study, shift it to be within bounds
            ts = np.array([float(t) for t in marksT.keys()])
            if float(thickness) < np.min(ts):
                value = ts.min()
            elif float(thickness) > np.max(ts):
                value = ts.max()
            logger.debug(f"Thickness marks: {ts} {thickness} {value}")

        except Exception as ex:
            logger.error(
                "Unable to determine precomputed thickness "
                f"and fillet radiu for {unitcell} ({form}): "
                f"{ex}"
            )
            marksT = None
    except Exception as ex:
        logger.error(
            "Unable to update tick markings based on " f"unitcell {unitcellForm}: {ex}"
        )
        marksT = None

    return marksT, value


@app.callback(
    [
        Output({"type": "card-L", "index": MATCH}, "value"),
        Output({"type": "card-W", "index": MATCH}, "value"),
        Output({"type": "card-H", "index": MATCH}, "value"),
        Output({"type": "card-AR", "index": MATCH}, "data"),
    ],
    [Input({"type": f"card-{t}", "index": MATCH}, "value") for t in ["L", "W", "H"]]
    + [Input({"type": f"card-ARlock", "index": MATCH}, "value")],
    [State({"type": f"card-AR", "index": MATCH}, "data")],
    prevent_initial_call=True,
)
def updateCardAR(L, W, H, lock, AR):
    # Get triggering context
    ctx = dash.callback_context
    triggered = ctx.triggered

    # Update the aspect ratio if the AR lock is activated
    if triggered and "card-ARlock" in triggered[0]["prop_id"]:
        dmin = min(L, W, H)
        AR = dict(L=L / dmin, W=W / dmin, H=H / dmin)
        logger.debug(f"Aspect ratio locked at {AR}")

        return dash.no_update, dash.no_update, dash.no_update, AR

    # If there is an aspect ratio lock, check to see if a dimension was updated.
    # If so, update the others accordingly.
    lockCards = [f"card-{dim}" for dim in "LWH"]
    if lock:
        logger.debug("Locked aspect ratios")
        # Loop through each triggering input. In theory, there should
        # only be one, as these are manual inputs
        for trig in ctx.triggered:
            tprop = parsePropID(trig["prop_id"])[0]
            logger.debug(f"Triggering property: {tprop}")
            if tprop["type"] in lockCards:
                # Pull out the driving dimension
                _, dim = tprop["type"].split("-")
                value = float(trig["value"])
                logger.debug(f"Dim {dim} updated to {value}")

                # Update the other dimensions according to the aspect ratio
                # scaling
                scaling = float(value) / AR[dim]
                logger.debug(f"Aspect ratio scaling: {scaling}")
                L, W, H = [AR[d] * scaling for d in "LWH"]

                # Make sure the values are still in bounds
                dmin = min(L, W, H)
                dmax = max(L, W, H)
                if dmin < 1:
                    L, W, H = L / dmin, W / dmin, H / dmin
                if dmax > 5:
                    L, W, H = [5 * d / dmax for d in [L, W, H]]

                # Update the aspect ratio definition
                dmin = min(L, W, H)
                AR = dict(L=L / dmin, W=W / dmin, H=H / dmin)
                logger.debug(f"New aspect ratio: {AR}")

                break

    return L, W, H, AR


@app.callback(
    Output({"type": "card-viz", "index": MATCH}, "children"),
    [
        Input({"type": f"card-{t}", "index": MATCH}, "value")
        for t in ["unitcell", "L", "W", "H", "T", "nx", "ny", "nz", "res"]
    ],
    prevent_initial_call=True,
)
def updateCardViz(unitcellForm, L, W, H, T, nx, ny, nz, res):
    try:
        split1 = unitcellForm.split(" (")
        unitcell = split1[0]
        form = split1[1][:-1].replace(" ", "")
        form = "graph" if form in ["truss", "corrugation"] else form

        logger.debug(f"Updating card vizualization for {unitcellForm}")
        # Note, smoothing radius is assumed to be the same as the
        # thickness. This is how the geometry was generated, but isn't
        # necessarily true in general.
        return _createSubCardViz(unitcell, L, W, H, T, T, form, nx, ny, nz, res)
    except:
        logger.debug("Missing input information")
        return []


@app.callback(
    Output({"type": "card-props", "index": MATCH}, "children"),
    [
        Input({"type": f"card-{t}", "index": MATCH}, "value")
        for t in ["unitcell", "L", "W", "H", "T"]
    ],
    [State("customOptions", "data"), State("customSurrogates", "data")],
    prevent_initial_call=True,
)
# @profile
def updateCardProps(unitcellForm, L, W, H, T, scustom, ssurrogate):
    try:
        split1 = unitcellForm.split(" (")
        unitcell = split1[0]
        form = split1[1][:-1].replace(" ", "").lower()

        logger.debug(f"Updating card properties for {unitcellForm}")
        # @BUG
        # Note, smoothing radius is assumed to be the same as the
        # thickness. This is how the geometry was generated, but isn't
        # necessarily true in general.
        return _createSubCardProps(unitcell, L, W, H, T, T, form, scustom, ssurrogate)
    except:
        logger.debug("Missing input information")
        return []


@app.callback(Output("faq-graph-definition", "children"), Input("faq-graph", "value"))
def updateFAQGraphDefinition(graph):
    # Output an empty Div is no graph unit cell is specified
    if not graph:
        return None

    # Pull out the stored graph definition
    definition = GRAPH_DEF[graph]

    # Create a table
    # Create table of properties
    try:
        # Define a table for the nodes
        header = [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Node number", style=dict(width="33%")),
                        html.Th("Coordinates (X/Y/Z)"),
                    ]
                )
            )
        ]

        rows = []
        for node, xyz in definition["node"].items():
            x, y, z = [xyz[k] for k in xyz.keys()]
            rows.append(html.Tr([html.Td(f"{int(node)}"), html.Td(f"({x}, {y}, {z})")]))

        tnodes = html.Div(
            [html.H3("Nodes"), dbc.Table(header + rows, bordered=True, striped=True)]
        )

        # Define a table for the beams
        header = [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Beam number", style=dict(width="33%")),
                        html.Th("Connecting nodes"),
                    ]
                )
            )
        ]

        rows = []
        for beam, nodes in definition["beam"].items():
            rows.append(
                html.Tr(
                    [html.Td(f"{int(beam)}"), html.Td(f"{nodes['n1']}/{nodes['n2']}")]
                )
            )

        if rows:
            tbeams = html.Div(
                [
                    html.Br(),
                    html.H3("Beam connectivity"),
                    dbc.Table(header + rows, bordered=True, striped=True),
                ]
            )
        else:
            tbeams = html.Div()

        # Define a table for the plate elements
        header = [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Plate number", style=dict(width="33%")),
                        html.Th("Connecting nodes"),
                    ]
                )
            )
        ]

        rows = []
        for face, nodes in definition["face"].items():
            rows.append(
                html.Tr(
                    [
                        html.Td(f"{int(face)}"),
                        html.Td("/".join([f"{nodes[k]}" for k in nodes.keys()])),
                    ]
                )
            )

        if rows:
            tplates = html.Div(
                [
                    html.Br(),
                    html.H3("Plate connectivity"),
                    dbc.Table(header + rows, bordered=True, striped=True),
                ]
            )
        else:
            tplates = html.Div()

        return [html.Br(), tnodes, tbeams, tplates]
    except Exception as ex:
        logger.debug(f"Unable to create graph geometry definition tables: {ex}")
        return None


@app.callback(
    Output("faq-walledtpms-definition", "children"), Input("faq-walledtpms", "value")
)
def updateFAQWalledTPMSDefinition(kind):
    # Output an empty Div is no unit cell is specified
    if not kind:
        return None

    # Create level set equation text
    try:
        # Pull out the stored definition
        definition = FAQ_WALLEDTPMS_OPTIONS[kind]["equation"]

        return [html.Br(), f"Φ(x, y, z) = {definition}", html.Br()]
    except Exception as ex:
        logger.debug(f"Unable to create walled TPMS equations: {ex}")
        return None
