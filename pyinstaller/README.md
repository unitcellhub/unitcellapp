UnitcellApp can currently be packaged into a standalone windows executable using the pyinstaller utility.

> [!NOTE]
> This executable is not intended for use on its own, but rather to be packaged using the Electron framework.
> See the electron folder for details.

To create the executable, first install the UV python packaging utility as found at https://docs.astral.sh/uv/getting-started/installation/.  
Navigate to the root unitcellapp directory and use the package manager uv to create the python environment based of off a controlled lock file:

```
uv sync --frozen
```

Then, run pyinstaller
```
uv run pyinstaller pyinstaller.spec
```

This will generate a windows executable and place it in dist/unitcellapp/unitcellapp.exe.
Running this executable starts a webserver running UnitcellApp, which can be accessed via a web browser by going to the url http://127.0.0.1:5030.
Note that this isn't a portable executable as it depends on the folder dist/unitcellapp/_internal.


