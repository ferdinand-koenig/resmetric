import argparse
import sys
import ast
import math

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
        # There is a bug with fig.show with which in approx. 10% of the cases the webpage would not load
        # Since some calculation may take long, the following approach is used with which the problem
        # seems to be solved
        fig.write_html('output.html', auto_open=True)


def print_workflow():
    workflow = u"""
                              Data                                                              
                                │                                                               
                                ▼                                                               
         ┌─────────────────────────────────────────────────────────┐                            
         │          Preprocessing                                  │                            
         │─────────────────────────────────────────────────────────│                            
         │ --smooth_criminal   Threshold-based value update filter │
         | --lin-reg THRSH  Find & use linear reg for steady states|
         │─────────────────────────────────────────────────────────│                            
         │ Not yet implemented:                                    │                            
         │ - Exponential Moving Average [EMA]                      │                            
         │ - Low-Pass Filters                                      │                            
         │ - Fourier Transformation                                │                            
         └─────────────────────────────────────────────────────────┘                            
                                │                                                               
                 ┌──────────────┴──────────────────────┐                                        
                 ▼                                     ▼                                        
     Dip-Agnostic Track [T-Ag]            Dip-Dependent Track [T-Dip]                           
                 │                                     ▼                                        
                 │                        ┌──────────────────────┐                              
                 │                        │     Detect Dips      │                              
                 │                        ├──────────────────────┤                              
                 │                        │ --max-dips           │                              
                 │                        │ --threshold-dips     │                              
                 │                        │ --manual-dips  MAN_D │                              
                 │                        │ --lin-reg-dips       │                              
                 │                        │                      │                              
                 │                        │ Not yet implemented: │                              
                 │                        │ --std [x sigma band] │                              
                 │                        └────────────┬─────────┘                              
                 ▼                                     ▼                                        
    ┌──────────────────────────────────┐ ┌───────────────────────────────────────────────────────┐ 
    │    Calculate Resilience          │ │   Calculate Resilience  (per Dip)                     │ 
    ├──────────────────────────────────┤ ├───────────────────────────────────────────────────────┤ 
    │ --auc    AUC devided by length   │ │ --dip-auc      AUC per dip (divided by length)        │ 
    │          of time frame and       │ │ --bars         Robustness, Recovery Level,            │ 
    │          weighted by kernel      │ │                and Recovery Rate                      │ 
    │ --count  How many times dropped  │ │ --irm          Integrated Resilience Metric (IRM)     │ 
    │          below threshold         │ └──────────────────────────┬────────────────────────────┘ 
    │ --time   How much time spent     │                            ▼                              
    │          below threshold         │ ┌───────────────────────────────────────────────────────┐ 
    │ --deriv  Show the 1st and 2nd    │ │         Calculate "Antifragility" /                   │ 
    │          derivatives             │ │         Resilience over Time                          │ 
    │                                  │ ├───────────────────────────────────────────────────────┤ 
    │ Advanced                         │ │ --calc-res-over-time    Calculate the differential    │ 
    │ --dips                           │ │                         quotient for every resilience │ 
    │ --drawdowns_traces               │ │                         metric of this track          │ 
    │ --drawdowns_shapes               │ │                                                       │ 
    │ --lin-reg THRESH (Cf. help page) │ │                                                       │
    └──────────────┬───────────────────┘ └──────────────────────────┬────────────────────────────┘ 
                   └────────────────────┬───────────────────────────┘                              
                                        ▼                                                          
                                Display or Save                                                    
                                                                                                                                                  
    """

    print("\nPreprocessing influences all subsequent steps. There are two tracks that can be executed independently: "
          "The Dip-Agnostic Track [T-Ag] and the Dip-Dependent Track [T-Dip]. In [T-Ag], resilience metrics do "
          "not depend on dips. In [T-Dip], all metrics are calculated w.r.t. a dip. Therefore, detecting dips "
          "is mandatory. Then metrics can be calculated, as well as how they change over time ('antifragility'). "
          "More options are available. See -h.")
    print(workflow)


def parse_manual_dips(dip_string):
    """Parse the manual dips input string into a list of tuples of integers, ensuring validity."""
    try:
        # Convert string to a list of tuples safely using `ast.literal_eval`
        dips = ast.literal_eval(dip_string)

        # Validate that it is a list of tuples of two non-negative integers
        if not isinstance(dips, list) or not all(isinstance(t, tuple) and len(t) == 2 for t in dips):
            raise ValueError

        # Check that all integers are non-negative
        if not all(isinstance(i, int) and i >= 0 for t in dips for i in t):
            raise ValueError("All integers in the tuples must be non-negative.")

        # Check that the tuples are in the correct order
        for i in range(len(dips) - 1):
            t0, t1 = dips[i]
            t2, t3 = dips[i + 1]
            if not (t0 > t1 >= t2 > t3):
                raise ValueError(f"Invalid order in tuples: {dips[i]} and {dips[i + 1]}.")

        return dips

    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(
            f"Invalid format for --manual-dips: {str(e)}. Expected a list of tuples, e.g., [(1, 2), (3, 4)].")


def main():
    parser = argparse.ArgumentParser(
        description='Generate and display or save a Plotly figure from JSON data with optional traces and analyses.')

    # Required argument
    parser.add_argument('json_file', nargs='?', type=str,
                        help='Path to the JSON file containing the data.')
    parser.add_argument('--workflow', action='store_true',
                        help='[Entry point for Beginners] Help-Command - Display the workflow diagram')

    preprocessing_group = parser.add_argument_group('Preprocessing Options')
    preprocessing_group.add_argument('--smooth_criminal', action='store_true',
                                     help='Smooth the series with a threshold-based value update filter '
                                          '(Hee-Hee! Ow!).')
    preprocessing_group.add_argument('--no-lin-reg-prepro', action='store_true',
                                     help='Using --lin-reg automatically acts as a preprocessor since it smoothes '
                                          'the steady states. With this flag, it can be turned it off.')

    agnostic_group = parser.add_argument_group('[T-Ag] Core Resilience-Related Trace Options')
    agnostic_group.add_argument('--auc', action='store_true',
                                help='Include AUC-related traces. '
                                     '(AUC devided by the length of time frame and different kernels applied '
                                     '[Uniform, Exponential decay, and inverse kernel])')
    agnostic_group.add_argument('--count', action='store_true',
                                help='Include traces that count dips below the threshold.')
    agnostic_group.add_argument('--time', action='store_true',
                                help='Include traces that accumulate time below the threshold.')
    agnostic_group.add_argument('--threshold', type=float, default=80,
                                help='Threshold for count and time traces in percent (default: 80).')
    agnostic_group.add_argument('--deriv', action='store_true',
                                    help='Include derivatives traces. [T-Ag]')

    # Create the [T-Dip] Dip Detection Algorithms argument group
    dip_detect_group = parser.add_argument_group('[T-Dip] Dip Detection Algorithms')
    dip_detect_group.add_argument(
        '--lin-reg',
        nargs='?',  # '?' means the argument can be passed without a value or with a single value
        const=True,  # This makes --lin-reg default to True if no value is provided
        default=False,  # By default, it's False if not provided at all
        help="Include linear regression traces. This is an algorithm that per se is dip-agnostic and therefore part "
             "of [T-Ag]. However, it can serve as a dip detection algorithm since the steady states before and after a "
             "disruption can be detected. To enable it, set --lin-reg-dips. "
             "A threshold can be provided until which steepness segments are "
             "kept. If only --lin-reg is given, threshold defaults to .5 percent. For analysis purposes, "
             "all segments can be kept by setting threshold to inf."
    )

    # Create the mutually exclusive group within the dip_detect_group
    mutually_exclusive_group = dip_detect_group.add_mutually_exclusive_group()

    mutually_exclusive_group.add_argument('--max-dips', action='store_true',
                                          help='Detect maximal dips based on local maxima (default)')
    mutually_exclusive_group.add_argument('--threshold-dips', action='store_true',
                                          help='Detect dips based on threshold (--threshold)')
    mutually_exclusive_group.add_argument('--manual-dips', type=parse_manual_dips,
                                          help='Manually specify dips as a list of tuples of integers, '
                                               'e.g., [(1, 2), (3, 4)].')
    mutually_exclusive_group.add_argument('--lin-reg-dips', action='store_true',
                                          help='Use linear regression segments of --lin-reg to define dips.')

    # Group for basic trace options
    basic_group = parser.add_argument_group('[T-Dip] Core Resilience-Related Trace Options')
    # basic_group.add_argument('--all-core', action='store_true',
    #                          help='Select all core resilience-related trace options.')
    basic_group.add_argument('--dip-auc', action='store_true',
                             help='Include AUC bars for the AUC of one maximal dip (AUC devided by the length of the '
                                  'time frame)')
    basic_group.add_argument('--bars', action='store_true',
                             help='Include bars for robustness, recovery time and recovery level.')
    # Create the mutually exclusive group within the dip_detect_group
    recovery_algorithm_group = basic_group.add_mutually_exclusive_group()

    recovery_algorithm_group.add_argument('--ada-ca', action='store_true', default=True,
                                          help='Adaptive capacity. Algorithm for calculating the recovery level. '
                                               'The first one is the ratio of new to prior steady state\'s value. '
                                               '(Q(t_ns) / Q(t_0)) (default)')
    recovery_algorithm_group.add_argument('--rec-ab', action='store_true',
                                          help='Recovery ability. Algorithm for calculating the recovery level. '
                                               'abs((Q(t_ns) - Q(t_r)) / (Q(t_0) - Q(t_r))) '
                                               'where Q(t_r) is the local minimum within a dip (Robustness).')

    basic_group.add_argument('--irm', action='store_true',
                             help='Include the Integrated Resilience Metric '
                                  '(Sansavini, https://doi.org/10.1007/978-94-024-1123-2_6, Chapter 6, formula 6.12, '
                                  'formula fixed to ((TAPL +1) ** -1)) cf. artefact publication))')

    anti_fragility_group = parser.add_argument_group('[T-Dip] Resilience-Related Metrics over Time Options ('
                                                     '"Anti-Fragility")')
    anti_fragility_group.add_argument('--calc-res-over-time', action='store_true',
                                      help='For every Core Resilience-Related Trace [T-Dip], '
                                           'calculate the differential quotient')

    fine_group = parser.add_argument_group('[Advanced][T-Ag] Fine-Grained Analysis Trace Options')
    fine_group.add_argument('--dips', action='store_true', help='Include all detected dips.')
    fine_group.add_argument('--drawdowns_traces', action='store_true',
                            help='Include the values of local drawdowns as traces.')
    fine_group.add_argument('--drawdowns_shapes', action='store_true',
                            help='Include the shapes of local drawdowns.')

    # Group for Advanced Settings
    settings_group = parser.add_argument_group('[Advanced] Settings')
    # Bayesian Optimization arguments
    settings_group.add_argument('--penalty_factor', type=float, default=0.05,
                                help='[--lr only] Penalty factor for the number of segments in Bayesian optimization '
                                     '(default: 0.05).')
    settings_group.add_argument('--dimensions', type=int, default=10,
                                help='[--lr only] Give the maximal number of segments for lin reg')
    # AUC-related arguments
    settings_group.add_argument('--weighted_auc_half_life', type=float, default=2,
                                help='[--auc only] Half-life for weighted AUC calculation (default: 2).')
    # Smoothing function arguments
    settings_group.add_argument('--smoother_threshold', type=float, default=2,
                                help='[--smooth-criminal only] Threshold for the smoothing function in percent ('
                                     'default: 2).')

    # Group for display or save options
    display_group = parser.add_argument_group('Display or Save Options')
    display_group.add_argument('--save', type=str, help='Path to save the figure as an HTML file.')
    display_group.add_argument('--silent', action='store_true',
                               help='Do not display the figure')

    args = parser.parse_args()

    # Handle the `--workflow` flag
    if args.workflow:
        print_workflow()
        return

    # Ensure that json_file is provided if `--workflow` is not used
    if args.json_file is None:
        print("\nError: The positional argument 'json_file' is required unless using the --workflow flag. "
              "If you encounter this error due to trying to pass --lin-reg without argument, "
              "please use the following expression: --lin-reg --\n"
              "See help page '-h'")
        sys.exit(1)

    dip_detect_algorithm = 'manual_dips' if args.manual_dips else 'threshold_dips' if args.threshold_dips \
        else 'max_dips' if args.max_dips else 'lin_reg_dips' if args.lin_reg_dips else None
    recovery_algorithm = 'recovery_ability' if args.rec_ab else 'adaptive_capacity'
    # if args.irm and not args.rec_ab:
    #     print('\n[Override notice] --irm requires --rec-ab as recovery algorithm.\n'
    #           'This sets --rec-ab and changes the algorithm. See -h for more info.')
    #     recovery_algorithm = 'recovery_ability'

    if args.lin_reg is not False:
        if isinstance(args.lin_reg, str) and args.lin_reg.lower() == 'inf':
                args.lin_reg = math.inf
        elif not (args.lin_reg is True):
            try:
                args.lin_reg = float(args.lin_reg)
            except ValueError:
                print(f"Invalid threshold value passed to --lin-reg: {args.lin_reg}. See -h")
                exit(1)
        print("This will take some minutes. You can grab a coffee ;) "
              "Or a beer, depending on the progress of your project.")
    else:
        if args.lin_reg_dips:
            print("If dips based on a linear regression should be used, linear regression must be calculated. "
                  "Provide the --lin-reg flag (optionally with threshold). See -h for help.")
            exit(1)

    # Convert args to a dictionary of keyword arguments
    kwargs = {
        'include_auc': args.auc,
        'include_dip_auc': args.dip_auc,
        'include_count_below_thresh': args.count,
        'include_time_below_thresh': args.time,
        'threshold': args.threshold,
        'include_draw_downs_traces': args.drawdowns_traces,
        'include_smooth_criminals': args.smooth_criminal,
        'include_dips': args.dips,
        'include_draw_downs_shapes': args.drawdowns_shapes,
        'include_bars': args.bars,
        'include_irm': args.irm,
        'include_derivatives': args.deriv,
        'include_lin_reg': args.lin_reg,
        'no_lin_reg_prepro': args.no_lin_reg_prepro,
        'penalty_factor': args.penalty_factor,
        'dimensions': args.dimensions,
        'weighted_auc_half_life': args.weighted_auc_half_life,
        'smoother_threshold': args.smoother_threshold,
        'calc_res_over_time': args.calc_res_over_time,
        'dip_detection_algorithm': dip_detect_algorithm,
        'manual_dips': args.manual_dips,
        'recovery_algorithm': recovery_algorithm,
    }

    plot_from_json_file(args.json_file, silent=args.silent, save_path=args.save, **kwargs)


if __name__ == '__main__':
    main()
