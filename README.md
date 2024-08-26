# ResMetric

`ResMetric` is a Python module designed to enhance Plotly figures
with resilience-related metrics. This comes in handy if you want
to explore different metrics and choose the best for your project.

The resilience of a system is especially of interest within the field of
self-adaptive systems where it measures the adaptability to shocks.
But there is still no standard metric.

This package enables users to add various optional traces and analyzes to
Plotly plots to visualize and analyze data more effectively. Therefore,
different resilience metrics can be explored. In addition,
the `metrics` submodule provides functions that can calculate the metrics individually!

## Key Features

- **Basic Trace Options**: Add AUC-related traces, count and time below threshold, drawdown shapes, smoothed series, detected dips, and more. This is where to start
- **Bayesian Optimization**: Optimize piecewise linear regression segments using Bayesian optimization. (Advanced)
- **Customizable Metrics**: Adjust parameters for AUC calculations and smoothing functions. (Advanced)
- **Use as Module or CLI**: Include the module in one of your projects or use the CLI to interact with the module!
- **Display or Save**: Display the plot in your browser or save it as an HTML file.

## Installation
### From wheel
Download the wheel (.whl) and
```bash
pip install resmetric-0.1-py3-none-any.whl
```

Distribution via PyPI might be available at some point. Stay tuned!

### From source
Go to the home directory and install via

```bash
pip install .
```

## Module Usage
### Importing the Module
To use the ResMetric module, import the `create_plot_from_data` function from the plot module:

```python
from resmetric.plot import create_plot_from_data
```

### Using the Function
The `create_plot_from_data` function generates a Plotly figure from JSON-encoded data and adds optional traces and analyses.

#### Function Signature
```python
def create_plot_from_data(json_str, **kwargs):
    """
    Generate a Plotly figure from JSON-encoded data with optional traces and analyses.

    Parameters:
    - json_str (str): JSON string containing the figure data.
    - **kwargs: Optional keyword arguments to include or exclude specific traces and analyses.

    Returns:
    - fig: Plotly Figure object with the specified traces and analyses included.
    """
```
#### Optional Keyword Arguments
##### Basic Trace Options
- **`include_auc`** (`bool`): Include AUC-related traces.
- **`include_count_below_thresh`** (`bool`): Include traces that count dips below the threshold.
- **`include_time_below_thresh`** (`bool`): Include traces that accumulate time below the threshold.
- **`threshold`** (`float`): Threshold for count and time traces (default: 80).
- **`include_draw_downs_traces`** (`bool`): Include traces representing the relative loss at each point in the time series, calculated as the difference between the current value and the highest value reached up to that point, divided by that highest value.
- **`include_smooth_criminals`** (`bool`): Include smoothed series (default threshold for smoothing: 2).
- **`include_dips`** (`bool`): Include detected dips.
- **`include_draw_downs_shapes`** (`bool`): Include shapes of local draw-downs.
- **`include_maximal_dips`** (`bool`): Include maximal dips, maximal draw-downs, and recoveries.
- **`include_bars`** (`bool`): Include bars for MDD and recovery.
- **`include_derivatives`** (`bool`): Include derivatives traces.
- **`include_lin_reg`** (`bool`): Include linear regression traces.

##### Bayesian Optimization (Only for `include_lin_reg`)
- **`penalty_factor`** (`float`): Penalty factor to penalize many segments / high number  of linear regression lines (default: 0.05).
- **`dimensions`** (`int`): Maximal number of segments for linear regression (default: 10).

##### AUC-related Options (Only for `include_auc`)
- **`weighted_auc_half_life`** (`float`): Half-life for weighted AUC calculation (default: 2).

##### Smoothing Function Options (Only for `include_smooth_criminals`)
- **`smoother_threshold`** (`float`): Threshold for the smoothing function in %. If a new point deviates only this much, the old value is preserved / copied (default: 2[%]).


## CLI Interface
The module also includes a command-line interface for quick usage.

### Command-Line Usage
```bash
resmetric-cli [options] json_file
```

The arguments are similar to the already described ones.
For more, see the help page:
```bash
resmetric-cli -h
```

## Examples
### Use as module
```python
import plotly.graph_objects as go
from resmetric.plot import create_plot_from_data

# Step 1: Create a Plotly figure
fig = go.Figure()  # Example: create a basic figure

# Customize the figure (e.g., add traces, update layout)
fig.add_trace(go.Scatter(x=[0, 1, 2], y=[.4, .5, .6], mode='lines+markers'))

# Get the JSON representation of the figure
json_data = fig.to_json()

# Step 2: Generate the plot with additional traces using the JSON data
fig = create_plot_from_data(
    json_data,
    include_count_below_thresh=True,
    include_maximal_dips=True,
    include_bars=True
)

# Display the plot
fig.show()

# Save the updated plot as HTML
fig.write_html('plot.html')

print("Figure saved as 'plot.html'")
```

### Use as CLI tool
#### Example 1 - AUC, count and time below threshold
```bash
resmetric-cli --count --time --auc .\example\fig.json
```
![count-time-auc.png](/example/count-time-auc.png)

#### Example 2 - Maximal dips with Maximal draw-downs and recoveries
```bash
 resmetric-cli --max_dips --bars .\example\fig.json
```
![max_dips-bars.png](/example/max_dips-bars.png)

#### Example 3 - Linear Regression with auto segmentation
```bash
 resmetric-cli --lg .\example\fig.json
```
![lg.png](/example/lg.png)

## Comment on execution times
The calculations for the linear regression (`--lg`) take some minutes.
So take a small break or fill up your coffee cup meanwhile :)

## Assumptions
Right now, the code has the assumptions
- that the provided data is always 0 &leq; data &leq; 1 and
- that the time index starts with 0 and increments by 1 unit with each step

## Demonstration cases
There are additional simple, synthetic cases to test the application in the folder `/evaluation/`.

## Contributing and developing
Contributions are welcome! Please submit a pull request or open an issue on the [GitHub repository](https://github.com/ferdinand-koenig/resmetric).

Please check the development guide (README.md) and scripts in `/development/`

## About
ResMetric is a Python module developed as part of my Master’s (MSc) study project in Computer Science at Humboldt University of Berlin. It is a project within the Faculty of Mathematics and Natural Sciences, Department of Computer Science, Chair of Software Engineering.

This project was supervised by Marc Carwehl. I would like to extend my sincere gratitude to Calum Imrie from the University of York for his invaluable support and feedback.

— Ferdinand Koenig

## License
Currently, all rights reserved. If interested, please contact me!
ferdinand (-at-) koenix.de

In the future, there might be a dual license approach: Commercial might require license fees, Non-commercial might be quite open
