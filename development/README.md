# Development Guide
In this folder (`development`), you can put development scripts.

## PEP8
Please follow PEP8 for code style.

## Separation of Concerns
- `cli.py` holds all and only holds logic for the CLI. It is a CLI for the Python module and therefore a CLI for `plot.py`
- `plot.py` is the driver that processes a request and puts together the plot. It does not perform resilience-related calculations but calls `metrics.py`
- `metrics.py` is the core that holds all calculations for resilience-related metrics

## Build a wheel
Make sure to update the version in `setup.py` according to semantic versioning (https://semver.org/)

```python setup.py clean --all bdist_wheel```
