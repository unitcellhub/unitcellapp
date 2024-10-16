"use strict";

// Based on https://github.com/matbloch/electron-flask

// Modules to control application life and create native browser window
const { app, BrowserWindow } = require('electron');
const path = require('path')
const EventEmitter = require('events')
const log = require('electron-log');

// Determine the backend based on the system architecture
// let backend;
// if (process.platform == 'win32') {
//     backend = path.resolve(process.cwd(), '../dist/unitcellapp/unitcellapp.exe')
// } else {
//     backend = path.resolve(process.cwd(), '../dist/unitcellapp/unitcellapp')
// }

// Keep a global reference of the mainWindowdow object, if you don't, the mainWindowdow will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow = null;
let subpy = null;
const loadingEvents = new EventEmitter()

// const PY_DIST_FOLDER = path.join(process.resourcesPath, 'dist/unitcellappp')
const PY_DIST_FOLDER = "../dist/unitcellapp"; // python distributable folder
const PY_SRC_FOLDER = "../"; // path to the python source
const PY_MODULE = "unitcellapp"; // the name of the main module

const createMainWindow = () => {
  // Create the browser mainWindow
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 1000,
    title: "UnitcellApp",
    // transparent: true, // transparent header bar
    icon: path.resolve(__dirname, "assets", "icon.png"),
    // fullscreen: true,
    // opacity:0.8,
    // darkTheme: true,
    // frame: false,
    resizeable: true,
  });

  // Display a loading window while we wait for the backend to load.
  // https://medium.com/red-buffer/how-to-create-a-splash-screen-for-electron-app-602b4da406d
  // https://github.com/ahmadsachal/electron_splash_screen
  mainWindow.loadFile('loading.html')


  // Wait until the backend server loads before loading the page.
  // https://interactiveknowledge.com/insights/create-electron-app-loading-screen
  loadingEvents.on('finished', () => {
    mainWindow.loadURL("http://localhost:5030/");
  })

  // Open the DevTools.
  //mainWindow.webContents.openDevTools();

  // Emitted when the mainWindow is closed.
  mainWindow.on("closed", function () {
    // Dereference the mainWindow object
    mainWindow = null;
  });
};

const isRunningInBundle = () => {
  return require("fs").existsSync(path.resolve(__dirname, PY_DIST_FOLDER));
};

const getPythonScriptPath = () => {
  if (!isRunningInBundle()) {
    return path.resolve(__dirname, PY_SRC_FOLDER, PY_MODULE + ".py");
  }
  if (process.platform === "win32") {
    return path.resolve(
      __dirname,
      PY_DIST_FOLDER,
      PY_MODULE + ".exe"
    );
  }
  return path.resolve(__dirname, PY_DIST_FOLDER, PY_MODULE);
};

const startPythonSubprocess = () => {
  let script = getPythonScriptPath();
  if (isRunningInBundle()) {
    console.log("Running bundled process: " + script)
    log.info("Running bundled process: " + script)
    subpy = require("child_process").execFile(script, []);
  } else {
    console.log("Running development script: " + script)
    subpy = require("child_process").spawn("python", [script]);
  }

  // Setup an event to trigger when the "INFO:waitress:Serving on"
  // is read in. Note that we need to listen to stderr rather than
  // stdout.
  // subpy.stdout.on('data', (data) => {
  //   let dataStr = `${data}`
  //   if (dataStr.includes("INFO:waitress:Serving on")){
  //     loadingEvents.emit("finished")
  //   }
  // })

  subpy.stderr.on('data', (data) => {
    let dataStr = `${data}`
    if (dataStr.includes("INFO:waitress:Serving on")){
      console.log("Background unitcellapp process successfully started.")
      log.info("Background unitcellapp process successfully started.")
      loadingEvents.emit("finished")

      // Once the process has been identified as started, turn off the listener
      subpy.stderr.removeAllListeners('data')
    } 
    // else {
    //   console.log("Error starting the background unitcellapp process: " + dataStr)
    //   log.info("Error starting the background unitcellapp process: " + dataStr)
    //   app.quit()
    // }
  })

};

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on("ready", function () {
  // start the backend server
  startPythonSubprocess();
  createMainWindow();
});

// disable menu
app.on("browser-window-created", function (e, window) {
  window.setMenu(null);
});

// Quit when all windows are closed.
app.on("window-all-closed", () => {
  // On macOS it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("activate", () => {
  // On macOS it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (subpy == null) {
    startPythonSubprocess();
  }
  if (win === null) {
    createMainWindow();
  }
});

app.on("quit", function () {
  // Stop background process
  subpy.kill();
});
