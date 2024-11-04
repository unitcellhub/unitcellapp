# Installation

## Setup

> [!NOTE]
> Before creating the electron app, you must first create an executable
> of the python code using pyinstaller. See the pyinstaller folder
> for more details. Once created, move on to the following steps.

Install yarn for your operating system

Move into the *electron* directory

```
cd electron
```

Install the package dependencies
```
yarn install --frozen-lockfile

```

## Run locally

To test out the Electron app without fully packaging it, run

```
yarn run debug
```

## Build and package

To create binary executable files and the corresponding installation files, run

```
yarn run dist 
```

which will output the created files in the dist/ folder.
Within this folder, the "UnitcellApp Setup X.X.X.exe" can be used to install UnitcellApp like any other windows program.



