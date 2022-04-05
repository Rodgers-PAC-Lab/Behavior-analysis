import datetime
import socket
import importlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import extras
import my.plot

importlib.reload(extras)

perf_metrics=pandas.read_pickle('perf_metrics', compression='infer')
acoustic_scored_by_n_ports = pandas.read_pickle('acoustic_scored_by_n_ports')
acoustic_scored_by_fraction_correct = pandas.read_pickle('acoustic_scored_by_fraction_correct')
session_df = pandas.read_pickle('session_df')
acoustic_trials= pandas.read_pickle('acoustic_trials')

mouse_names = [
    'M1_PAFT', 'M3_PAFT',
    'M2_PAFT', 'M4_PAFT',
    'F1_PAFT', 'F3_PAFT',
    'F2_PAFT', 'F4_PAFT',
    ]

cohort = {
    'all': [
        'M1_PAFT', 'M3_PAFT',
        'M2_PAFT', 'M4_PAFT',
        'F1_PAFT', 'F3_PAFT',
        'F2_PAFT', 'F4_PAFT',
        ],
}
np.random.seed(0)  # if you don't like it, change it
universal_colorbar = np.random.permutation(my.plot.generate_colorbar(len(mouse_names)))
universal_colorbar = pandas.DataFrame(
    universal_colorbar, index=mouse_names)
#Figure 1 plot- basic stats
#List the things we want to plot
plot_names = ['fc', 'rcp', 'n_trials']
fc_data = perf_metrics['fc']
fc_data=fc_data.unstack('mouse')

# Choose dates to plot
all_dates = sorted(perf_metrics.index.levels[1])

# Choose recent dates (for perf plot), and the mice that are in them
recent_dates = all_dates[-18:]

#creates a figure 'f' made of several plots. Makes as many subplots as there are in the 'plot names' list.
#It'll be one column of subplots
f, axa = plt.subplots(len(plot_names), 1, figsize=(5, 8), sharex=False)
f.subplots_adjust(hspace=.9, bottom=.1, top=.95)

for ax, plot_name in zip(axa, plot_names):
    # Get data (corresponding column)
    data = perf_metrics[plot_name]
    #print(data)

    # Put mouse in columns, date and data in rows
    data = data.unstack('mouse')
    data = data.reset_index(drop=True)
    # Plot
    for colname in data.columns:
        try:
            color = universal_colorbar.loc[colname].values
        except KeyError:
            color = 'k'
        ax.plot(data[colname], '.-', color=color, label=colname)

        # Pretty
        ax.set_title(plot_name)
        my.plot.despine(ax)
        # Lims and chance bar
        if plot_name == 'fc':
            ax.set_ylim((0, 1))
            ax.plot(ax.get_xlim(), [1 / 7., 1 / 7.], 'k--', lw=.75)
        elif plot_name == 'rcp':
            ax.set_ylim((4, 0))
            ax.plot(ax.get_xlim(), [3., 3.], 'k--', lw=.75)
        elif plot_name == 'n_trials':
            ax.set_ylim(ymin=0)

        xt = np.array(list(range(len(recent_dates))))
        xtl = [dt.strftime('%m-%d') for dt in recent_dates]
        ax.set_xticks(xt)
        ax.set_xticklabels(xtl, rotation=45)
        ax.set_xlim((xt[0] - .5, xt[-1] + .5))
    continue