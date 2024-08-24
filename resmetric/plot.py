import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import pwlf
from .metrics import (
    detect_peaks,
    _get_dips,
    extract_max_dips,
    extract_mdd_from_dip,
    get_recovery,
    calculate_auc,
    calculate_weighted_auc,
    calculate_inverse_weighted_auc,
    _half_life_to_lambda,
    time_below_threshold,
    count_dibs_below_threshold_series,
    calculate_max_drawdown,
    _find_next_smaller,
    smoother,
    _perform_bayesian_optimization,
    _make_color_pale_hex
)

def create_plot_from_data(json_str, **kwargs):
    """
    Generate a Plotly figure from a JSON-encoded Plotly figure with optional
    traces and analyses based on existing traces.

    Parameters
    ----------
    json_str : str
        JSON string containing the figure data.
    **kwargs : dict, optional
        Optional keyword arguments to include or exclude specific traces and
        analyses:

        - include_auc (bool): Include AUC-related traces.
        - include_count_below_thresh (bool): Include traces counting dips below
          the threshold.
        - include_time_below_thresh (bool): Include traces accumulating time
          below the threshold.
        - threshold (float): Threshold for count and time traces (default is 80).
        - include_draw_downs_traces (bool): Include traces representing the
          relative loss at each point in the time series, calculated as the
          difference between the current value and the highest value reached
          up to that point, divided by that highest value.
        - include_smooth_criminals (bool): Include smoothed series.
          (Hee-Hee! Ow!)
        - include_dips (bool): Include detected dips.
        - include_draw_downs_shapes (bool): Include shapes of local draw-downs.
        - include_maximal_dips (bool): Include maximal dips, maximal draw-downs,
          and recoveries.
        - include_bars (bool): Include bars for MDD and recovery.
        - include_derivatives (bool): Include derivatives traces.
        - include_lin_reg (bool): Include linear regression traces.
        - penalty_factor (float): Penalty factor for Bayesian Optimization
          (default is 0.05).
        - dimensions (int): Dimensions for Bayesian Optimization (default is 10)
          (Max. N. of segments for lin. reg.).
        - weighted_auc_half_life (float): Half-life for weighted AUC calculation
          (default is 2).
        - smoother_threshold (int): Threshold for smoother function
          (default is 2).

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        Plotly Figure object with the specified traces and analyses included.
    """
    # Convert JSON string to Plotly figure
    fig = pio.from_json(json_str)

    # Extract data series and initialize trace lists
    series = fig.data
    smooth_criminals = []
    auc_traces = []
    time_below_thresh_traces = []
    count_below_thresh_traces = []
    threshold_line = []
    drawdown_traces = []
    draw_down_shapes = []
    maximal_dips_shapes = []
    maximal_dips_bars = []
    dips_horizontal_shapes = []
    derivative_traces = []
    lin_reg_traces = []

    # Retrieve optional arguments with defaults
    threshold = kwargs.get('threshold', 80)
    penalty_factor = kwargs.get('penalty_factor', 0.05)
    dimensions = kwargs.get('dimensions', 10)
    weighted_auc_half_life = kwargs.get('weighted_auc_half_life', 2)
    smoother_threshold = kwargs.get('smoother_threshold', 2)

    # Initialize variables for global x limits
    global_x_min = float('inf')
    global_x_max = float('-inf')

    for i, s in enumerate(series):
        y_values = s.y
        x_values = s.x if s.x is not None else np.arange(len(y_values))  # Assuming x-values are indices

        # Update global x limits
        global_x_min = min(global_x_min, x_values[0])
        global_x_max = max(global_x_max, x_values[-1])

        # Peak detection and dip extraction
        maxs = detect_peaks(np.array(y_values))
        mins = detect_peaks(-np.array(y_values))
        y_mins = np.array(y_values)[mins]
        dips = _get_dips(y_values, maxs=maxs)
        max_dips = extract_max_dips(dips)
        mdd_info = extract_mdd_from_dip(max_dips, mins, y_values)

        # Compute 1st and 2nd derivatives
        first_derivative = np.gradient(y_values)
        second_derivative = np.gradient(first_derivative)

        # Append derivative traces if requested
        if kwargs.get('include_derivatives'):
            derivative_traces.extend([
                go.Scatter(
                    y=first_derivative,
                    mode='lines+markers',
                    marker=dict(symbol='diamond-dot', color=fig.layout.template.layout.colorway[i]),
                    name=f'1st derivative - {s.name}',
                    legendgroup=f'1st derivative - {s.name}',
                ),
                go.Scatter(
                    y=second_derivative,
                    mode='lines+markers',
                    marker=dict(symbol='diamond-dot', color=fig.layout.template.layout.colorway[i]),
                    line=dict(dash='dot'),
                    name=f'2nd derivative - {s.name}',
                    legendgroup=f'2nd derivative - {s.name}',
                )
            ])

        # Fit the piecewise linear model and add to traces if requested
        if kwargs.get('include_lin_reg'):
            # Perform Bayesian Optimization to find the optimal number of segments
            optimal_segments = _perform_bayesian_optimization(x_values, y_values,
                                                              penalty_factor=penalty_factor, dimensions=dimensions)
            pwlf_model = pwlf.PiecewiseLinFit(x_values, y_values)
            pwlf_model.fit(optimal_segments)
            y_hat = pwlf_model.predict(x_values)
            lin_reg_traces.append(
                go.Scatter(
                    x=x_values,
                    y=y_hat,
                    mode='lines',
                    line=dict(color=fig.layout.template.layout.colorway[i]),
                    name=f'PWLF ({optimal_segments} Segments) - {s.name}',
                    legendgroup=f'PWLF - {s.name}'
                )
            )

        # Append traces for dips and drawdowns if requested
        if kwargs.get('include_dips'):
            for b, e in dips:
                dips_horizontal_shapes.append(
                    go.Scatter(
                        x=[b, e],
                        y=[y_values[b], y_values[b]],
                        mode='lines',
                        line=dict(dash='dot', color=fig.layout.template.layout.colorway[i]),
                        name=f'Dip Line {s.name}',
                        legendgroup=f'Dip Line {s.name}'
                    )
                )
            dips_horizontal_shapes.append(
                go.Scatter(
                    x=[e for _, e in dips],
                    y=[y_values[s] for s, _ in dips],
                    mode='markers',
                    marker=dict(symbol='x', color=fig.layout.template.layout.colorway[i]),
                    name=f'Dips {s.name}',
                    legendgroup=f'Dip Line {s.name}',
                ))

        if kwargs.get('include_draw_downs_shapes'):
            for mini in mins:
                next_smaller = _find_next_smaller(maxs, mini)
                if next_smaller is None:
                    continue
                draw_down_shapes.append(
                    go.Scatter(
                        x=[mini, mini],
                        y=[y_values[mini], y_values[next_smaller]],
                        mode='lines',
                        line=dict(dash='dot', color=fig.layout.template.layout.colorway[i], width=2),
                        name=f'Draw Downs - {s.name}',
                        legendgroup=f'Draw Downs - {s.name}',
                    )
                )

        if kwargs.get('include_maximal_dips'):
            for max_dip, info in mdd_info.items():
                maximal_dips_shapes.append(
                    go.Scatter(
                        x=[max_dip[0], max_dip[1]],
                        y=[y_values[max_dip[0]], y_values[max_dip[0]]],
                        mode='lines',
                        line=dict(dash='dot', color=fig.layout.template.layout.colorway[i]),
                        name=f'Max Dips - {s.name}',
                        legendgroup=f'Max Dips + MDD + rel. Rec - {s.name}'
                    )
                )
                maximal_dips_shapes.append(
                    go.Scatter(
                        x=[max_dip[1]],
                        y=[y_values[max_dip[0]]],
                        mode='markers',
                        marker=dict(symbol='x', color=fig.layout.template.layout.colorway[i]),
                        name=f'Max Dips - {s.name}',
                        legendgroup=f'Max Dips + MDD + rel. Rec - {s.name}'
                    )
                )
                maximal_dips_shapes.append(
                    go.Scatter(
                        x=[info['line'][0][0], info['line'][1][0]],
                        y=[info['line'][0][1], info['line'][1][1]],
                        mode='lines',
                        line=dict(color=fig.layout.template.layout.colorway[i], width=2, dash='dash'),
                        name=f'Max Drawdown {s.name}',
                        legendgroup=f'Max Dips + MDD + rel. Rec - {s.name}'
                    )
                )
                maximal_dips_bars.append(
                    go.Bar(
                        x=[info['line'][0][0]],
                        y=[info['value']],
                        width=1,
                        marker=dict(color=fig.layout.template.layout.colorway[i]),
                        opacity=0.25,
                        name=f'MDD + rel. Rec Bars - {s.name}',
                        legendgroup=f'MDD + rel. Rec Bars - {s.name}',
                    )
                )

            recovery_info = get_recovery(y_values, max_dips)
            if kwargs.get('include_bars'):
                for e, recovery in recovery_info.items():
                    maximal_dips_bars.append(
                        go.Bar(
                            x=[recovery['line'][0][0]],
                            y=[recovery['relative_recovery']],
                            width=1,
                            marker=dict(color=fig.layout.template.layout.colorway[i]),
                            opacity=0.25,
                            name=f'Rel. Recovery - {s.name}',
                            legendgroup=f'MDD + rel. Rec Bars - {s.name}',
                        )
                    )
                    maximal_dips_shapes.append(
                        go.Scatter(
                            x=[recovery['line'][0][0], recovery['line'][1][0]],
                            y=[recovery['line'][0][1], recovery['line'][1][1]],
                            mode='lines',
                            line=dict(dash='dot', color=fig.layout.template.layout.colorway[i]),
                            name=f'Recovery Line {s.name}',
                            legendgroup=f'Max Dips + MDD + rel. Rec - {s.name}',
                        )
                    )

        # Append smooth criminal traces if requested
        if kwargs.get('include_smooth_criminals'):
            smooth_criminals.append(go.Scatter(
                name=f"Smoothed {s.name}",
                y=smoother(list(y_values), threshold=smoother_threshold),
                mode='lines+markers',
                marker=dict(color=fig.layout.template.layout.colorway[i]),
                legendgroup=f'Smoothed {s.name}'
            ))

        # Append AUC-related traces if requested
        if kwargs.get('include_auc'):
            auc_values = calculate_auc(y_values)
            auc_traces.append(go.Scatter(
                name=f"AUC {s.name}",
                legendgroup=f"AUC {s.name}",
                y=auc_values,
                mode='lines',
                marker=dict(color=fig.layout.template.layout.colorway[i])
            ))

            auc_values_exp = calculate_weighted_auc(y_values, _half_life_to_lambda(weighted_auc_half_life))
            auc_traces.append(go.Scatter(
                name=f"AUC-exp {s.name}",
                legendgroup=f"AUC-exp {s.name}",
                y=auc_values_exp,
                mode='lines',
                marker=dict(color=fig.layout.template.layout.colorway[i])
            ))

            auc_values_inv = calculate_inverse_weighted_auc(y_values)
            auc_traces.append(go.Scatter(
                name=f"AUC-inv {s.name}",
                legendgroup=f"AUC-inv {s.name}",
                y=auc_values_inv,
                mode='lines',
                marker=dict(color=fig.layout.template.layout.colorway[i])
            ))

        # Append count and time traces if requested
        if kwargs.get('include_count_below_thresh'):
            count_below_thresh_traces.append(go.Scatter(
                y=count_dibs_below_threshold_series(y_values, threshold),
                name=f"Count below {threshold}% - {s.name}",
                legendgroup=f"Count below {threshold}% - {s.name}",
                marker=dict(color=fig.layout.template.layout.colorway[i]),
                yaxis='y3'
            ))

        if kwargs.get('include_time_below_thresh'):
            time_below_thresh_traces.append(go.Scatter(
                y=time_below_threshold(y_values, threshold),
                name=f"Time below {threshold}% - {s.name}",
                legendgroup=f"Time below {threshold}% - {s.name}",
                marker=dict(color=fig.layout.template.layout.colorway[i]),
                yaxis='y2'
            ))

        # Append drawdown traces if requested
        if kwargs.get('include_draw_downs_traces'):
            drawdown_traces.append(go.Scatter(
                y=calculate_max_drawdown(y_values)[1],
                name=f"Drawdown Trace - {s.name}",
                legendgroup=f"Drawdown Trace- {s.name}",
                marker=dict(color=fig.layout.template.layout.colorway[i]),
                yaxis='y3'
            ))

        # Update the original series with a pale color
        s.update(
            mode='lines+markers',
            marker=dict(color=_make_color_pale_hex(fig.layout.template.layout.colorway[i])),
            line=dict(color=_make_color_pale_hex(fig.layout.template.layout.colorway[i]))
        )

    # Include threshold line if requested
    if kwargs.get('include_time_below_thresh') or kwargs.get('include_count_below_thresh'):
        threshold_line.append(go.Scatter(
            x=[global_x_min, global_x_max],  # Extend the line across the global x-axis range
            y=[threshold/100, threshold/100],
            mode='lines',
            name=f'Threshold: {threshold}%',
            line=dict(dash='dash', color='red')
        ))

    # Update layout to include secondary y-axes
    fig.update_layout(
        yaxis2=dict(
            title='Time below Threshold',
            overlaying='y',
            side='right'
        ),
        yaxis3=dict(
            title='Dip below Threshold Count',
            overlaying='y',
            anchor='free',
            side='right',
            autoshift=True
        ),
        yaxis4=dict(
            title='Drawdown (%)',
            overlaying='y',
            anchor='free',
            side='right',
            autoshift=True
        )
    )

    # Combine all requested traces
    all_traces = list(series)
    if kwargs.get('include_auc'):
        all_traces += auc_traces
    if kwargs.get('include_count_below_thresh'):
        all_traces += count_below_thresh_traces
    if kwargs.get('include_time_below_thresh'):
        all_traces += time_below_thresh_traces
    if kwargs.get('include_time_below_thresh') or kwargs.get('include_count_below_thresh'):
        all_traces += threshold_line
    if kwargs.get('include_draw_downs_traces'):
        all_traces += drawdown_traces
    if kwargs.get('include_smooth_criminals'):
        all_traces += smooth_criminals
    if kwargs.get('include_dips'):
        all_traces += dips_horizontal_shapes
    if kwargs.get('include_draw_downs_shapes'):
        all_traces += draw_down_shapes
    if kwargs.get('include_maximal_dips'):
        all_traces += maximal_dips_shapes
    if kwargs.get('include_bars'):
        all_traces += maximal_dips_bars
    if kwargs.get('include_derivatives'):
        all_traces += derivative_traces
    if kwargs.get('include_lin_reg'):
        all_traces += lin_reg_traces

    # Update the figure with new data and layout
    fig = go.Figure(data=all_traces, layout=fig.layout)
    fig.update_layout(barmode='overlay',margin=dict(l=10, r=20, t=10, b=10), legend=dict(
        x=.02,
        y=.98,
        xanchor='left',
        yanchor='top'
    ))

    # Hide all traces initially and only show the first trace in each legend group
    legend_groups_seen = set()
    for trace in fig.data:
        legend_group = trace.legendgroup
        if legend_group:
            if legend_group not in legend_groups_seen:
                trace.update(showlegend=True)
                legend_groups_seen.add(legend_group)
            else:
                trace.update(showlegend=False)
        else:
            trace.update(showlegend=True)

    return fig
