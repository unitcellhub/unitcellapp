from pathlib import Path
import dill as pickle
import blosc
import logging
import tables
import re
import numpy as np
from PIL import Image
import io
import base64
from copy import copy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from functools import partial
import pandas as pd
from unitcellapp.options import _OPTIONS
from typing import List

# Create a logger
logger = logging.getLogger(__file__)

# Create a cache folder
CACHE = Path(__file__).parent / Path("cache/")
if not CACHE.exists():
    CACHE.mkdir()

# Custom forward scaling for surrogate model
def yforward(x):
    return x ** (1 / 3.0)

# Custom backward scaling for surrogate model
def ybackward(x):
    return x**3

# Construct the lattice cache filename
def cacheFilename(kind: str, form: str, unitcell: str) -> Path:
    return CACHE / Path(f"{kind}_{form}_{unitcell}.pkl")

# Load cache from file
def cacheLoad(kind, form, unitcell):
    # Read blosc compressed pickle file
    filename = cacheFilename(kind, form, unitcell)
    logger.debug(f"Loading cache file {filename}")
    with open(filename, "rb") as f:
        data = f.read()
    return pickle.loads(blosc.decompress(data))

# Save cache file
def cacheSave(data, kind, form, unitcell) -> Path:
    # Write cached files
    tosave = blosc.compress(
        pickle.dumps(data)
    )
    filename = cacheFilename(kind, form, unitcell)
    with open(filename, "wb") as f:
        f.write(tosave)
    
    return filename

# Convert image data to encoded image string
def image2base64(mat):
    im = Image.fromarray(mat[0])
    buffer = io.BytesIO()
    im.save(buffer, format="png")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/png;base64, " + encoded_image
    return im_url


def cacheCreate(database: Path|str|None=None, kinds: List[str]=["data", "images", "surrogates"]) -> None:
    """ Create a new cache of properties from the database file 

    Parameters
    ----------
    database: Path, str, or None (default=None)
        If specified, defines the location of the reference .h5 database file (for example unitcelldb.h5 from github.com/unitcellhub/unitcelldb). If not, assumes the file is in the root directory under database/unitcelldb.h5.
    kinds: list of "data", "images", "surrogates" (default=["data", "images", "surrogates"])
        Defines the properties to cache. Note that this is an 

    """

    logging.basicConfig()
    logger.setLevel(logging.DEBUG)

    # Define the default database file if one isn't specified
    if not database:
        database = Path(__file__).parent.parent.parent / Path("database/unitcelldb.h5")
    else:
        database = Path(database)
    logger.debug(f"Using database file {database}")

    # Check that the database exists. If not, through an error pointing to the github repository
    if not database.exists():
        raise RuntimeError(f"Unable to find lattice database file {database.absolute()}. "
                            "Check that it is in this location. To get a current "
                            "version of the database, clone UnitcellDB from "
                            "github.com/unitcellhub/unitcelldb and put unitcelldb.h5 "
                            "into a folder called 'database' at the root of unitcellapp.")


    # Add in dataprocessing for image thumbnail storage
    OPTIONS_MOD = copy(_OPTIONS)
    OPTIONS_MOD["image"] = dict(
        name="Unitcell thumbnail",
        info="",
        ref=["image"],
        calc=image2base64,
    )

    # Load the lattice database file and fit surrogate models. Additionally, case
    # the results to speedup the launch process
    if "data" in kinds or "images" in kinds:
        with tables.open_file(database, "r") as h5file:
            logger.debug("Reading and pre-processing database...")
            # Pull out the database table
            table = h5file.root.design

            # Pull out the data and store the data
            _DATA = dict(graph={}, walledtpms={})
            for r in table.iterrows():
                unitcell = r["unitcell"].decode("utf-8")
                form = r["form"].decode("utf-8")
                # subdata = [r[col] for col in columns]
                # Loop through each QOI option and run the "calc" function in the row data.
                subdata = [
                    v["calc"](list(map(r.__getitem__, v["ref"])))
                    for v in OPTIONS_MOD.values()
                ]
                try:
                    _DATA[form][unitcell].append(subdata)
                except:
                    _DATA[form][unitcell] = [subdata]

        # Write cached files
        for form, unitcells in _DATA.items():
            for unitcell in unitcells:
                images = [d[-1] for d in _DATA[form][unitcell]]
                data = [d[:-1] for d in _DATA[form][unitcell]]
                if "data" in kinds: cacheSave(data, "data", form, unitcell)
                if "images" in kinds: cacheSave(images, "images", form, unitcell)
    else:
        logger.debug("Using existing data cache files to create surrogate models.")
        _DATA = {}
        for cache in CACHE.glob("data_*.pkl"):
            _, form, unitcell = cache.stem.split("_")
            _data = cacheLoad("data", form, unitcell)
            try:
                # Note, we need to append on a file column to be consistent with the above data structure
                _DATA[form][unitcell] = _data
            except KeyError:
                # Initialize dictionaries as they don't exist yet 
                _DATA[form] = {}
                # Store data
                _DATA[form][unitcell] = _data

    
    # Check if surrogate models are desired. If not, exit early
    if not "surrogates" in kinds:        
        logger.debug("Surrogate model caches weren't request. Existing early.")
        return

    # Define y forward and backward scaling functions
    regStiffness = re.compile("(E|G|k)((\d){1,2}|min|max)")
    regVonMises = re.compile("vonMisesWorst\d{2}")

    # Fit surrogate model to each unitcell dataset
    _SURROGATE = dict(graph={}, walledtpms={})
    columns = list(_OPTIONS.keys())
    logger.debug("Fitting surrogate models to data...")
    for form, unitcells in _DATA.items():
        for unitcell in unitcells.keys():
            # Create panda's data frame with unit cell data
            try:
                # Data read in from database that has an extra column at the end for the image thumbnail
                df = pd.DataFrame(
                    np.array([r[:-1] for r in _DATA[form][unitcell]]), columns=columns
                )
            except:
                # Cache data loaded

                print(np.array(_DATA[form][unitcell]).shape, len(columns))
                df = pd.DataFrame(
                    np.array(_DATA[form][unitcell]), columns=columns
                )


            # Check to see if surrogate model information was already
            # loaded for this geometry. If so, assume it is the same as
            # the cached value and move on to the next geometry. If not,
            # generate the surrogate model for this geometry.
            # NOTE: This is just to save time when simply updating a few
            # geometries. This can be commented out if a full redefine
            # is requried.
            try:
                _SURROGATE[form][unitcell]
                continue
            except KeyError:
                _SURROGATE[form][unitcell] = {}

            params = ["length", "width", "height", "thickness", "radius"]
            MAPPING_AR = {"x": "length", "y": "width", "z": "height"}
            for param in df.columns:
                # Skip independent variables
                if param in params or "custom" in param:
                    continue

                # Certain geometric properties can be explicitly defined
                # with a simple mathematical formulation, such as aspect ratio.
                # Define these relationships rather than fitting them.
                if "AR" in param:
                    
                    dir1 = MAPPING_AR[param[0].lower()]
                    dir2 = MAPPING_AR[param[1].lower()]

                    # An expression-based model can be defined with a list with the following
                    # definition.
                    # The 1st is a list of strings defining the custom equation variables and
                    # the 2nd is a string defining the custom equation.
                    
                    # Store model, scaling functions, and scores
                    _SURROGATE[form][unitcell][param] = {
                        "model": [[dir1, dir2], f"{dir1}/{dir2}"],
                        "xscaler": None,
                        "yscaler": None,
                        "scores": None,
                    }
                    continue

                # Create the kernel for the Gaussian Process Regression
                kernel = 1 * RBF(
                    length_scale=1, length_scale_bounds=(1e-4, 1e6)
                ) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-8, 1e3))

                # Create the Gaussian Processes Regressor
                # Note, pre transforming the data With a log function
                # drastically improves the regression.
                gpr = GaussianProcessRegressor(
                    kernel=kernel, alpha=1e-8, n_restarts_optimizer=0, normalize_y=True
                )
                xscaler = FunctionTransformer(
                    np.log, inverse_func=np.exp, validate=True, check_inverse=True
                )

                # Create new model
                model = make_pipeline(xscaler, gpr)

                if (
                    regVonMises.match(param)
                    or regStiffness.match(param)
                    or "anisotropyIndex" in param
                ):
                    # Scaling the y data for stiffness and stress values
                    # helps clean up the fits. This is some partial
                    # rationale here in that this is roughly how these
                    # parameters scale with relative density, but that
                    # should already be captured with the independent
                    # variable scaling. This really just helps with the
                    # large order of magnitude shifts within these data
                    # types.
                    yscaler = FunctionTransformer(
                        yforward,
                        inverse_func=ybackward,
                        validate=True,
                        check_inverse=True,
                    )
                else:
                    # Default transformer is the identify transform
                    yscaler = FunctionTransformer()

                # Define training data
                Xn = np.array(df[params])
                Yn = yscaler.fit_transform(np.array(df[param])[:, None])[:, 0]

                # Define convenience inverse scaling function to be
                # stored with the output
                def iyscaling(sy):
                    sy = np.array(sy)
                    return yscaler.inverse_transform(sy[:, None])[:, 0]

                # Run a quick cross-validation to make sure there are
                # not inherent issues with the model fit
                logger.debug(f"Cross validating {unitcell} ({form}) {param}")
                scores = cross_val_score(model, Xn, Yn, cv=5)
                if scores.mean() < 0.85:
                    logger.warning(
                        "Poor model validation scores of "
                        f"{scores} for "
                        f"{unitcell} ({form}) {param}. "
                        "Check input data and/or the "
                        "input model to determine the issue."
                    )

                # Run model fit
                logger.debug(f"Fitting model for {unitcell} ({form}) {param}")
                model.fit(Xn, Yn)

                # Store model, scaling functions, and scores
                _SURROGATE[form][unitcell][param] = {
                    "model": model,
                    "xscaler": xscaler,
                    "yscaler": yscaler,
                    "scores": scores,
                }

            # Write cached files
            cacheSave(_SURROGATE[form][unitcell], "surrogates", form, unitcell)
    logger.debug("Complete.")
    #         # break

if __name__ == "__main__":
    # cacheCreate(kinds=['surrogates'])
    cacheCreate()





# if __name__ == "__main__":
#     for cache in CACHE.glob("database_*.pkl"):
#         # Parse the old naming convention
#         _, form, unitcell = cache.stem.split("_")
#
#         # Load the current cache data
#         current = cacheLoad("database", form, unitcell)
#         
#         # Pull the images out of the data cache
#         _data = current['data']
#         images = [d[-1] for d in _data]
#         data = [d[:-2] for d in _data]
#
#
#         # Split the data into 3 separate files for faster access
#         cacheSave(current['surrogate'], 'surrogates', form, unitcell)
#         cacheSave(images, "images", form, unitcell)
#         cacheSave(data, "data", form, unitcell)

