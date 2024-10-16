# UnitcellEngine

The majority of the analysis toolkit -- UnitcellEngine -- works without any special configuration. Simply run

```
python setup.py install
```

or, if using the package manager pip, navigate the base folder and run

```
pip install .
```

## Prerequisites
UnitcellEngine only supporst Python 3. Due to python version limitations on a few of UnitcellEngine's dependencies (numba in particular), Python 3.10 is the latest version that is supported.


## Windows machines
If running on a Windows machine, you may sometimes need to first install Microsoft Visual C++ for some of the dependencies to install correctly (such as number and scikit-learn). See https://visualstudio.microsoft.com/visual-cpp-build-tools/ for details on how to install the Windows C++ Build Tools.

## High Performance Computing and remote Linux servers
When running on remote systems, there are a few features that don't work straight out of the box. If geometry rendering is required (which relies on VTK), additional utitilies will likely need to be installed.

### CENTOS and RH systems without root privelages
Xvfb is an X server that can run on machines with no display hardware and no physical input devices. It emulates a dumb framebuffer using virtual memory. Install xvfb following the steps defined here: https://stackoverflow.com/questions/36651091/how-to-install-packages-in-linux-centos-without-root-user-with-automatic-depen. Note, on most HPC systems, these files should be installed on a high performance filesystem rather than a network drive. 

In addition to xvfb, you will need to install the OpenGL libraries "mesa-libGL" and "mesa-libGL-devel" using the same steps as for xvfb. In some cases, you might still get an "GLSL 1.50 is not supported. Supported versions are: 1.10, 1.20, 1.30, 1.00 ES, and 3.00 ES" error. In this case, add the following environmental variable

```
export MESA_GL_VERSION_OVERRIDE=3.2
```


# UnitcellApp

UnitcellApp can be packaged as a desktop application using a combination of pyinstaller and Electron. To do so, run "build.bat" followed by "make.bat".

