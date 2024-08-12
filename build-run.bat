@REM This is a development script that removes the packages and installs from source.
@REM Adapt the last line for the case you want to test during development.
pip uninstall -y resmetric
pip install .
resmetric-cli --count ".\example\fig.json"
