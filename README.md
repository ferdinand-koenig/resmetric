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
- **Seperation of Dip-Agnostic ([T-Ag]) and Dip-Dependent Metrics ([T-Dip])**: Analyze your performance graph w.r.t. resilience and 'antifragility' (See *Getting Started* Section)
- **Dip-Agnostic Metrics** do not depend on where a disruption starts and the system is fully recovered. (AUC, count and time below threshold, etc)
- **Dip-Dependent Metrics** need the definition / detection where a disruption starts and when it is recovered. (AUC per dip, robustness, recovery, recovery rate, Intergrated Resilience Metric)
- **Customizable Metrics**: Adjust parameters for AUC calculations, smoothing functions, and use Bayesian optimization to fit linear regressions to detect steady states and therefore dips. (Advanced)
- **Use as Module or CLI**: Include the module in one of your projects or use the CLI to interact with the module!
- **Display or Save**: Display the plot in your browser or save it as an HTML file.

## Installation and Setup
The following guide is for beginners. If you are not a beginner, feel free to skip the installation instructions!

If you need the wheel file, you can find it in the [releases section](https://github.com/ferdinand-koenig/resmetric/releases).

Having trouble installing it? Please check the installation note in the [appendix](#appendix) of this README.

### Step-by-Step Installation and Setup
Go to a folder where you want to place the installation and download the wheel (`.whl` file)

0. **Get Python** (Only if you do not have Python `>= 3.8`)
   
   You should download and install Python from [download](https://www.python.org/downloads/). Go for the latest version, but everything `>= 3.8` will be fine.
   A guide for installation and version check can be found [here](https://realpython.com/installing-python/).
   
1. **Install the package**
   Create a virtual environment and install the package in it:
    - **Linux or MacOS:**
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        python3 -m pip install --upgrade pip
        python3 -m pip install resmetric-1.0.0-py3-none-any.whl
        ```
    - **Windows:**
        ```cmd
        py -m venv .venv
        .venv\Scripts\activate
        py -m pip install --upgrade pip
        py -m pip install resmetric-1.0.0-py3-none-any.whl
        ```
2. **Use the package**
   If you continue right after step 1, you can use the tool and jump right in.
   If you have installed the package previously, you have to source your shell again:

   - For Linux or MacOS:
     ```bash
     source .venv/bin/activate
     ```

   - For Windows:
     ```cmd
     .venv\Scripts\activate
     ```
     
   Now, explore the tool on your own! You can try using the [CLI](##CLI-Interface), the [module in Python](##Module-Usage),
   or check out our [CLI Examples](##Use-as-CLI-tool) to see demonstrations and replicate them.

   
3. **After Use**
   To deactivate the virtual environment, run
   ```bash
   deactivate
   ```
4. **Uninstall**
   If you have created the virtual environment only for this project (as you did if you followed this little guide),
   just delete the `.venv` directory. That's it!
   ```bash
   rm -r .venv
   ```
   You could also uninstall Python if you no longer need it.


## Getting started
Familiarize yourself with the workflow. It will help to understand how the dip-agnostic track and dip-dependent track
differ and how to calculate 'antifragility' (resilience over time)
```bash
resmetric-cli --workflow
```

Then, try the examples (below).

## Module Usage
### Importing the Module
To use the ResMetric module, import the `create_plot_from_data` function from the plot module:

```python
from resmetric.plot import create_plot_from_data
# or
import resmetric.plot as rp
# rp.create_plot_from_data(...)
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
##### Preprocessing
- **`include_smooth_criminals`** (`bool`): Include smoothed series (see `smoother_threshold`).

##### [T-Ag] Dip-Agnostic Resilience Trace Options
- **`include_auc`** (`bool`): Include AUC-related traces. (AUC divided by the length of the time frame and
          different kernels applied)
- **`include_count_below_thresh`** (`bool`): Include traces that count dips below the threshold.
- **`include_time_below_thresh`** (`bool`): Include traces that accumulate time below the threshold.
- **`threshold`** (`float`): Threshold for count and time traces (default: 80).
- **`include_dips`** (`bool`): Include detected dips.
- **`include_draw_downs_traces`** (`bool`): Include traces representing the relative loss at each point in the time series, calculated as the difference between the current value and the highest value reached up to that point, divided by that highest value.
- **`include_draw_downs_shapes`** (`bool`): Include shapes of local draw-downs.
- **`include_derivatives`** (`bool`): Include derivatives traces.

##### [T-Dip] Dip-Dependent Resilience Trace Options
- **`dip_detection_algorithm`** (`str`): Specifies the dip detection algorithm to use.
  It can be 'max_dips' (default), 'threshold_dips', 'manual_dips', 'lin_reg_dips' (the last requires
          `include_lin_reg`)
- **`manual_dips`** (`list of tuples` or `None`): If 'manual_dips' is selected as the dip detection algorithm,
  this should be a list of tuples specifying the manual dips. (E.g., `[(2,5), (33,42)]` for two dips from time t=2 to 5 and
  33 to 42
- **`include_lin_reg`** (`bool` or `float`): Include linear regression traces. Optionally float for threshold of slope. 
  Slopes above the absolute value are discarded. The threshold defaults to 0.5% (for value set to True).
  It is possible to pass `math.inf`. See also `no_lin_reg_prepro`
- **`no_lin_reg_prepro`** (`bool`): `include_lin_reg` automatically preprocesses and updates the series. If you do
          not wish this, set this flag to True

- **`include_max_dip_auc`** (`bool`): Include AUC bars for the AUC of one maximal dip
  (AUC devided by the length of the time frame)
- **`include_bars`** (`bool`): Include bars for MDD and recovery.
- **`include_gr`** (`bool`): Include the Integrated Resilience Metric
  (cf. Sansavini, https://doi.org/10.1007/978-94-024-1123-2_6, Chapter 6, formula 6.12).
  Requires kwarg `recovery_algorithm='recovery_ability'`.
- **`recovery_algorithm`** (`str` or `None`): Decides the recovery algorithm. Can either be `adaptive_capacity` (default)
  or `recovery_ability`. The first one is the ratio of new to prior steady state's value `(Q(t_ns) / Q(t_0))`.
  The last one is `abs((Q(t_ns) - Q(t_r))/ (Q(t_0) - Q(t_r)))`
  where `Q(t_r)` is the local minimum within a dip (Robustness).

- **`calc_res_over_time`** (`bool`): Calculate the differential quotient for every per-dip Resilience-Related Trace.

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
import resmetric.plot as rp

# Step 1: Create a Plotly figure
fig = go.Figure()  # Example: create a basic figure

# Customize the figure (e.g., add traces, update layout)
# Here you add the data of the performance graph where you want to investigate resilience (over time)
fig.add_trace(go.Scatter(x=[0, 1, 2], y=[.4, .5, .6], mode='lines+markers', name='Q1'))
fig.add_trace(go.Scatter(x=[0, 1, 2], y=[.4, .3, .6], mode='lines+markers', name='Q2'))

# Get the JSON representation of the figure
json_data = fig.to_json()

# Step 2: Generate the plot with additional traces using the JSON data
rp.set_verbose(False)  # disable prints (only relevant for lin reg)
fig = rp.create_plot_from_data(
    json_data,
    include_count_below_thresh=True,
    include_maximal_dips=True,
    include_bars=True
)

# Display the plot
# fig.show()
# There is a bug with fig.show with which in approx. 10% of the cases the webpage would not load
# Since some calculation may take long, the following approach is used with which the problem
# seems to be solved.
fig.write_html('output.html', auto_open=True)

# Save the updated plot as HTML
fig.write_html('plot.html')

print("Figure saved as 'plot.html'")
```

### Use as CLI tool
*Note: The wheel (`.whl`) does not include example material.*

To get started, you can download the example data `fig.json` [here](https://github.com/ferdinand-koenig/resmetric/blob/main/example/fig.json). 
Please place the downloaded file in a subdirectory called `example` to use the examples as provided.

To illustrate the tool's capabilities with relevant data, we utilized the classification accuracy graph presented by Gheibi and Weyns[^1]. Although the graph is not included in their paper, it reflects the same reported results and can be accessed through the accompanying replication package[^2]. We added a fourth curve by incorporating the classification performance of an ensemble learner, complementing the three existing performance curves. Their study proposes a self-adaptive, lifelong machine learning model, demonstrating its effectiveness in classifying gas in a pipeline[^3][^4].

[^1]: [Omid Gheibi and Danny Weyns. 2022. Lifelong Self-Adaptation: Self-Adaptation Meets Lifelong Machine Learning. In 17th International Symposium on Software Engineering for Adaptive and Self-Managing Systems (SEAMS ’22), May 18–23, 2022, PITTSBURGH, PA, USA. ACM, New York, NY, USA, 12 pages](https://doi.org/10.1145/3524844.3528052)
[^2]: [Replication package for Gas Delivery Case](https://people.cs.kuleuven.be/~danny.weyns/software/LLSAS/) Accessed: 2024-05-07
[^3]: [Vergara, A., Vembu, S., Ayhan, T., Ryan, M. A., Homer, M. L., and Huerta, R. Chemical gas sensor drift compensation using classifier ensembles. Sensors and Actuators B: Chemical 166-167 (2012), 320–329.](https://doi.org/10.1016/j.snb.2012.01.074)
[^4]: [Data: Gas Sensor Array Drift at Different Concentrations](https://doi.org/10.24432/C5MK6M)


#### Example 1 - AUC, count and time below threshold
AUC gives three traces per trace in the original plot: The weighted moving average,
weighted 1) uniformly 2) with an exponential decay and 3) with an inverse distance weighting.
With the latter two, more recent points contribute more to the average AUC at a given time.
```bash
resmetric-cli --count --time --auc ./example/fig.json
```
![count-time-auc.png](/example/count-time-auc.png)

#### Example 2 - Maximal dips with Maximal draw-downs and recoveries
```bash
 resmetric-cli --max_dips --bars ./example/fig.json
```
![max_dips-bars.png](/example/max_dips-bars.png)

#### Example 3 - Linear Regression with auto segmentation
```bash
 resmetric-cli --lin-reg -- ./example/fig.json
```
![lg.png](/example/lg.png)

## Comment on execution times
The calculations for the linear regression (`--lin-reg --`) take some minutes.
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
ResMetric is a Python module developed as part of my Master’s (M.Sc.) study project in Computer Science at Humboldt University of Berlin. It is a project within the Faculty of Mathematics and Natural Sciences, Department of Computer Science, Chair of Software Engineering.

This project was supervised by Marc Carwehl. I would like to extend my sincere gratitude to Calum Imrie from the University of York for his invaluable support and feedback.

— Ferdinand Koenig

## License
Currently, all rights reserved. If interested, please contact me!
ferdinand (-at-) koenix.de

In the future, there might be a dual license approach: Commercial might require license fees, Non-commercial might be quite open


## Appendix
### Note: Installation on Ubuntu 23.04+, Debian 12+, and Similar OSs (`error: externally-managed-environment`)

Starting with Ubuntu 23.04, Debian 12, and similar systems, some operating systems have implemented
[PEP 668](https://peps.python.org/pep-0668/) to help protect system-managed Python packages from being overwritten or
broken by `pip`-installed packages. This means that attempting to install Python packages globally
outside of a virtual environment can lead to errors such as `error: externally-managed-environment`.

**This is not a bug** with the `resmetric` package but rather expected behavior enforced by the operating system to keep
system packages stable. Generally, to install Python packages without issues, you must use a virtual environment.

If you encounter the installation error mentioning PEP 668 or `error: externally-managed-environment`, follow one of the strategies below:

#### Option 1: Use the [step-by-step installation guide](#step-by-step-installation-and-setup)
The virtual environment fixes this issue.

#### Option 2: Use `pipx`
1. **Install and setup `pipx`**
   ```bash
   sudo apt-get install -y pipx
   pipx ensurepath
    ```
2. **Install package via `pipx`**
   Replace `pip` by `pipx` in the installation command. This may look like
   ```bash
   pipx install resmetric-1.0.0-py3-none-any.whl
   ```
   Make sure to use the right filename for the wheel file.
3. **You are all set up!**


