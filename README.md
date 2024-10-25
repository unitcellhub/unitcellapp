# UnitcellApp

*UnitcellApp* is the Graphical User Interface (GUI) of the *UnitcellHub* software suite.
It integrates the vast database of lattice properties in *UnitcellDB*, the geometry and visualization engine of *UnitcellEngine*, and a elements of Machine Learning (ML) to enable a user to select an ideal lattice geometry for a given engineering application.
The primary interface is built upon Ploty's open source Dash platform, which creates a Flask-based web application.

## Installation
To enable broad usage, a number of deployment frameworks exists depending on a user's need.
Overall, there are three primary frameworks:

- Native server web application run on a local or remote machine
- Docker wrapped application, which can be run locally or distributed through a service like DockerHub or MyBinder
- Electron-based desktop application (which is currently only support on Windows)

<a id="native"></a>
### Native web application (Windows, Mac, Linux)

Fundamentally a Flask-based web application, this is the most obvious distribution methodology.
The minimum requirements are to have python3 and git installed locally on your machine (be that Windows, Linux, or Mac).
To install locally, the recommended methodology is to run the application within a local python virtual environment.
In a terminal prompt

```
git lfs install
git clone https://github.com/unitcellhub/unitcellapp
cd unitcellapp
git checkout main
python -m venv .venv
```

On Windows, activate the environment with
```
.venv/Scripts/active
```
One Linux or Mac, activate the environment with
```
source .venv/bin/activate
```

Then, install the required dependencies with
```
python -m pip install -r requirements.txt

```

To run the web application, run


```
python unitcellapp.py

```

Then, open a web browsers and go to the url [http://127.0.0.1:5030](http://127.0.0.1:5030).

Installation on a remote server is somewhat system dependent.
The basic setup previously described for local execution should be followed, with the exception of how the software is launched.
To start the web server, a WSGI HTTP server like gunicorn is required.

### Docker application (Mac, Linux, Windows)

The Docker method acts as a more generalizable deployment mechanism.
To use this option, first install [docker](https://www.docker.com/).
Once installed, the current release of *UnitcellApp* can be pulled from [DockerHub](https://hub.docker.com/) by running

```
docker pull unitcellhub/unitcell:latest
```

If a custom docker image is desired, first clone the respository, make the desired modifications, and then run the following

```
docker build --tag unitcellhub/unitcellapp .
```

To run *UnitcellApp* using the docker implementation, run

```
docker run --env PORT=5030 --publish 5030:5030 --terminate unitcellhub/unitcellapp
```
and then access *UnitcellApp* in your preferred browser at http://127.0.0.1:5030.

> [!NOTE]
> Port 5030 is arbitrary and can be changed as desired.

### Electron desktop application (Windows only)

> [!WARNING]
> This is a beta feature with no guarantees.
> On Windows, it often requires a degree of debugging to get working.
> This workflow should in theory work on other operating systems, but hasn't been tested.

*UnitcellApp* can be wrapped by Electron to create a desktop application.
As the backbone features are built upon Python, this requires some initial setup to create the binaries required by Electron.
To do so, open a command line and run

```
pyinstaller.bat
```

which can take 5-20 minutes to run.
This batch files runs pyinstaller to create an executable version of *UnitcellApp*, which is placed in the folder "dist/unitcellapp/unitcellapp.exe" 
Due to limitations with pyinstaller, this is a somewhat unstable process.
It have been verified to work on Windows.
It can likely be generalized to Mac and Linux, but hasn't been completed successfully to-date.

Next, make the Electron app by running

```
electron.bat
```

which can take 5-30 minutes to run.
Once complete, the Windows installer can be found in "electron/dist/UnitcellApp Setup X.X.X.exe" (where the Xs denote the current version number).
Note that this is currently the least stable deployment mechanism.

## Custom lattice databases

The default implementation is built around the data in [*UnitcellDB*](https://github.com/unitcellhub/unitcelldb).
This data has been post processed and cached as [Dill](https://github.com/uqfoundation/dill)-based pickle files.
A custom dataset, however, can be generated using [*UnitcellEngine*](https://github.com/unitcellhub/unitcellengine) and then fed into *UnitcellApp*.
- To do so, run all the desired *UnitcellEngine* simulations and compbine them together to create a primary HDF5 database file "database.h5".
- Place this file in the local *UnitcellApp* folder tree under "database/database.h5".
- Clear the "dashboard/cache" folder of any files names "*.pkl".
- We now need to post-process this database, not only calculating quantities of interested, but also running the ML learning process. This is all computed the first time that *UnitcellApp* is run (which can take 30-60 minutes depending on the size of the database) and then cached for subsequent access. To do so, run *UnitcellApp* as a [native web application](#native).

Now, anytime that *UnitcellApp* is run, it will use this custom database.

