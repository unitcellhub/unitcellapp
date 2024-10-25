from pathlib import Path
import dill as pickle
import blosc
import logging

# Create a logger
logger = logging.getLogger(__file__)

# Create a cache folder
CACHE = Path(__file__).parent / Path("cache/")
if not CACHE.exists():
    CACHE.mkdir()

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

if __name__ == "__main__":
    for cache in CACHE.glob("database_*.pkl"):
        # Parse the old naming convention
        _, form, unitcell = cache.stem.split("_")

        # Load the current cache data
        current = cacheLoad("database", form, unitcell)
        
        # Pull the images out of the data cache
        _data = current['data']
        images = [d[-1] for d in _data]
        data = [d[:-2] for d in _data]


        # Split the data into 3 separate files for faster access
        cacheSave(current['surrogate'], 'surrogates', form, unitcell)
        cacheSave(images, "images", form, unitcell)
        cacheSave(data, "data", form, unitcell)
        # break
