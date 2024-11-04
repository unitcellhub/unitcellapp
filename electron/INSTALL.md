# Installation

## Setup

> [!NOTE]
> Before create the electron app, you must first create an executable
> of the python code using pyinstaller. See the pyinstaller folder
> for more details. Once created, move on to the following steps.

Install yarn for your operating system

Move into the *electron* directory

```
cd electron
```

Install the package dependencies
```
yarn install

```

## Run locally

```
yarn run start
```

## Build and package

To create binary executable files, run

```
yarn run dist 
```

which will output the created files in the dist/ folder


