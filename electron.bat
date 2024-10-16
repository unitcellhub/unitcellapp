@ECHO OFF
:: Build electron app

:: Move to the electron directory and make the application
cd electron
yarn install
yarn run dist
cd ..