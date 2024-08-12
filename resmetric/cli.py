import argparse
from .plot import create_plot_from_data


def plot_from_json_file(json_file, silent=False, save_path=None, **kwargs):
    """
    Reads JSON data from a file, generates a Plotly figure with the specified options,
    and either displays the figure, saves it to a file, or both.

    Parameters:
    - json_file (str): Path to the JSON file containing the Plotly figure data.
    - silent (bool): If True, only save the figure and do not display it.
    - save_path (str, optional): Path to save the figure as a static HTML file.
    - **kwargs: Optional keyword arguments to control trace inclusion and analysis.
    """
    with open(json_file, 'r') as f:
        data = f.read()
    fig = create_plot_from_data(data, **kwargs)

    if save_path:
        fig.write_html(save_path)
        print(f"Figure saved to {save_path}")

    if not silent:
        fig.show()


def main():
    parser = argparse.ArgumentParser(
        description='Generate and display or save a Plotly figure from JSON data with optional traces and analyses.')

    # Required argument
    parser.add_argument('json_file', type=str, help='Path to the JSON file containing the data.')

    # Group for basic trace options
    basic_group = parser.add_argument_group('Basic Trace Options')
    basic_group.add_argument('--auc', action='store_true', help='Include AUC-related traces.')
    basic_group.add_argument('--count', action='store_true',
                             help='Include traces that count dips below the threshold.')
    basic_group.add_argument('--time', action='store_true',
                             help='Include traces that accumulate time below the threshold.')
    basic_group.add_argument('--threshold', type=float, default=80,
                             help='Threshold for count and time traces (default: 80).')
    basic_group.add_argument('--drawdowns_traces', action='store_true',
                             help='Include the values of local drawdowns as traces.')
    basic_group.add_argument('--drawdowns_shapes', action='store_true',
                             help='Include the shapes of local draw-downs.')
    basic_group.add_argument('--smooth_criminal', action='store_true',
                             help='Include smoothed series (Hee-Hee! Ow!).')
    basic_group.add_argument('--dips', action='store_true', help='Include detected dips.')
    basic_group.add_argument('--max_dips', action='store_true',
                             help='Include maximal dips, maximal draw-downs, and recoveries')
    basic_group.add_argument('--bars', action='store_true', help='Include bars for MDD and recovery.')
    basic_group.add_argument('--deriv', action='store_true', help='Include derivatives traces.')
    basic_group.add_argument('--lg', action='store_true', help='Include linear regression traces.')

    # Group for Bayesian Optimization arguments
    bayesian_group = parser.add_argument_group('Bayesian Optimization (Only for --lg)')
    bayesian_group.add_argument('--penalty_factor', type=float, default=0.05,
                                help='Penalty factor for the number of segments in Bayesian optimization (default: '
                                     '0.05).')
    bayesian_group.add_argument('--dimensions', type=int, default=10,
                                help='Give the maximal number of segments for lin reg')

    # Group for AUC-related arguments
    auc_group = parser.add_argument_group('AUC-related Options (Only for --auc)')
    auc_group.add_argument('--weighted_auc_half_life', type=float, default=2,
                           help='Half-life for weighted AUC calculation (default: 2).')

    # Group for Smoothing function arguments
    smooth_group = parser.add_argument_group('Smoothing Function Options (Only for --smooth-criminal)')
    smooth_group.add_argument('--smoother_threshold', type=float, default=2,
                              help='Threshold for the smoothing function (default: 2).')

    # Group for display or save options
    display_group = parser.add_argument_group('Display or Save Options')
    display_group.add_argument('--save', type=str, help='Path to save the figure as an HTML file.')
    display_group.add_argument('--silent', action='store_true',
                               help='Do not display the figure')

    args = parser.parse_args()

    # Convert args to a dictionary of keyword arguments
    kwargs = {
        'include_auc': args.auc,
        'include_count_below_thresh': args.count,
        'include_time_below_thresh': args.time,
        'threshold': args.threshold,
        'include_draw_downs_traces': args.drawdowns_traces,
        'include_smooth_criminals': args.smooth_criminal,
        'include_dips': args.dips,
        'include_draw_downs_shapes': args.drawdowns_shapes,
        'include_maximal_dips': args.max_dips,
        'include_bars': args.bars,
        'include_derivatives': args.deriv,
        'include_lin_reg': args.lg,
        'penalty_factor': args.penalty_factor,
        'dimensions': args.dimensions,
        'weighted_auc_half_life': args.weighted_auc_half_life,
        'smoother_threshold': args.smoother_threshold
    }

    plot_from_json_file(args.json_file, silent=args.silent, save_path=args.save, **kwargs)


if __name__ == '__main__':
    main()
