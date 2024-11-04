import json
import sys
from pathlib import Path

# Add the version directory to the search path if it isn't already there
VERSION = Path(__file__).parent.parent / "_version.py"
sys.path.append(VERSION.parent.as_posix())

# Import the current version version number
try:
    from unitcellapp import __version__ as version
except ModuleNotFoundError:
    raise RuntimeError(
        "Could not find the '_version.py' in the src/unitcellapp package diretory. "
        "To update the electron package version number, UnitcellApp must first be built. "
    )

# Open the current npm package file and update the version number
PACKAGE = Path(__file__).parent / "package.json"
with open(PACKAGE, "r") as f:
    package = json.load(f)

# Update the version number and write to file
if "Unknown" in version:
    version = "0.0.0.dev"
else:
    # Yarn is strict on the formatting of a version. This isn't an issue
    # for a standard release, but causes issues for dev builds. Here,
    # we switch from a .devXXXXX trailing reference to a -devXXXXXX.
    split = version.split(".")
    version = ".".join(split[:3])
    if len(split) > 3:
        version += f"-{split[-1]}"
package["version"] = version
with open(PACKAGE, "w") as f:
    json.dump(package, f, indent=4)
