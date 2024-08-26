@REM This is a development script that removes the packages and installs from source.
@REM Adapt the last line for the case you want to test during development.
pip uninstall -y resmetric
pip install .
resmetric-cli --max_dips --bars --max-dip-auc --calc-res-over-time ".\example\fig.json"
