# UnitcellApp

*UnitcellApp* is the Graphical User Interface (GUI) of the *UnitcellHub* software suite.
It integrates the vast database of lattice properties in *UnitcellDB*, the geometry and visualization engine of *UnitcellEngine*, and a elements of Machine Learning (ML) to enable a user to select an ideal lattice geometry for a given engineering application.
The primary interface is built upon Ploty's open source Dash platform, which creates a Flask-based web application.

## Public web application
Try *UnitcellApp* without the need to install software at [www.unitcellapp.org](https://www.unitcellapp.org). 

## Installation
To enable broad usage, a number of deployment frameworks exists depending on a user's need.
Overall, there are three primary frameworks:

- Native server web application run on a local or remote machine
- Docker wrapped application, which can be run locally or distributed through a service like DockerHub or MyBinder. Pre-build images are hosted on ghcr.io.
- Electron-based desktop application (which is currently only supported on Windows). Pre-build executables are hosted in on the [releases](https://github.com/unitcellhub/unitcellapp/releases) page of the GitHub repository.

<a id="native"></a>
### Native web application (Windows, Mac, Linux)

Fundamentally a Flask-based web application, this is the most obvious distribution methodology.
The minimum requirements are to have [git](https://git-scm.com/) and [uv](https://docs.astral.sh/uv/) installed locally on your machine (be that Windows, Linux, or Mac); followed the relevant installation procedures for your operating system before moving forward.
First, clone the github repository

```
git lfs install
git clone https://github.com/unitcellhub/unitcellapp
cd unitcellapp
git checkout main
```

The *uv* python package manager is used to ensure consistent application builds and is simply run by 

```
uv run production
```

From here, navigate to the url [http://127.0.0.1:5030](http://127.0.0.1:5030) in your preferred web browser to access the application.

For developers, make any desired changes to the code and run in debug mode by executing

```
uv run debug
```

In this mode, any updates that are made in the source code force the application to reinitialize with the given changes.
In general, it is best to debug code through the addition of relevant logging outputs while running the above command rather than a conventional debugger.


### Docker application (Mac, Linux, Windows)

The Docker method acts as a more generalizable deployment mechanism.
To use this option, first install [docker](https://www.docker.com/).
Once installed, the current release of *UnitcellApp* can be pulled from Github's docker registry ghcr.io by running

```
docker pull ghcr.io/unitcellhub/unitcellapp:latest
```

You can also specify specific versions of *UnitcellApp* rather than the "latest" tag. For example:

```
docker pull ghcr.io/unitcellhub/unitcellapp:0.0.11
```

If a custom docker image is desired, first clone the repository, make the desired modifications, and then run the following

```
docker build --tag unitcellhub/unitcellapp:custom .
```

To run *UnitcellApp* using the docker implementation, run

```
docker run --env PORT=5030 --publish 5030:5030 --terminate unitcellhub/unitcellapp:<tag>
```
and then access *UnitcellApp* in your preferred browser at http://127.0.0.1:5030.
Here, <tag> refers to the version of *UnitcellApp* that has been pulled or built (such as "latest", "0.0.11", or "custom" as defined the the previous examples.)

> [!NOTE]
> Port 5030 is arbitrary and can be changed as desired.

### Electron desktop application (Windows only)

> [!WARNING]
> This is a beta feature with no guarantees.
> On Windows, it often requires a degree of debugging to get working.
> This workflow should in theory work on other operating systems, but hasn't been tested.

*UnitcellApp* had been wrapped by Electron to create a desktop application.
For pre-built executables, see the [releases](https://github.com/unitcellhub/unitcellapp/releases) page.
Note that the pre-build executables aren't Code Signed; so, you will be warned when installing the software.

If you are concerned with application security due to a lack of code signing or want to build your own distribution with custom features, the executable can be build locally.
As the backbone features are built upon Python, this requires some initial setup to create the binaries required by Electron.
To create the executable, first install the UV python packaging utility as found at https://docs.astral.sh/uv/getting-started/installation/.  
Then, run

```
pyinstaller.bat
```

which can take 2-10 minutes to run.
This batch files runs pyinstaller to create an executable version of *UnitcellApp*, which is placed in the folder "dist/unitcellapp/unitcellapp.exe" 
Due to limitations with pyinstaller, this is a somewhat unstable process.
It has been verified to work on Windows.
It can likely be generalized to Mac and Linux, but hasn't been completed successfully to-date.

Next, make the Electron app by running

```
electron.bat
```

which can take 2-10 minutes to run.
Once complete, the Windows installer can be found in "electron/dist/UnitcellApp Setup X.X.X.exe" (where the Xs denote the current version number).
Note that this is currently the least stable deployment mechanism.

## Custom lattice databases

The default implementation is built around the data in [*UnitcellDB*](https://github.com/unitcellhub/unitcelldb).
This data has been post processed and cached as [Dill](https://github.com/uqfoundation/dill)-based pickle files.
A custom dataset, however, can be generated using [*UnitcellEngine*](https://github.com/unitcellhub/unitcellengine) and then fed into *UnitcellApp*.
- To do so, run all the desired *UnitcellEngine* simulations and combine them together to create a primary HDF5 database file "database.h5".
- Place this file in the local *UnitcellApp* folder tree under "database/database.h5".
- Clear the "dashboard/cache" folder of any files names "*.pkl".
- Build the *UnitcellApp* python environment as defined in [native web application](#native) section.
- We now need to post-process this database, not only calculating quantities of interested, but also running the ML learning process. This is done by running the cache module with the command: python src/unitcellapp/cache.py. Depending on the size of the data base, this can take a long time to execute; for example, it takes about 30-60 min to process the UnitcellDB database. Once completed, simply run *UnitcellApp* as a [native web application](#native) (or build a docker image for more portability).


Now, anytime that *UnitcellApp* is run, it will use this custom database.

