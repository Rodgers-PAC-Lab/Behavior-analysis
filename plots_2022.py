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
# Generate color bar
np.random.seed(0) # if you don't like it, change it
universal_colorbar = np.random.permutation(my.plot.generate_colorbar(len(mouse_names)))
universal_colorbar = pandas.DataFrame(
    universal_colorbar, index=mouse_names)

#Figure 1 plot- basic stats
#List the things we want to plot
plot_names = ['fc', 'rcp', 'n_trials']

#creates a figure 'f' made of several plots. Makes as many subplots as there are in the 'plot names' list.
#It'll be one column of subplots
f, axa = plt.subplots(len(plot_names), 1, figsize=(5, 8), sharex=False)
f.subplots_adjust(hspace=.9, bottom=.1, top=.95)

for ax, plot_name in zip(axa, plot_names):
    # Get data (corresponding column)
    data = perf_metrics[plot_name]
    print(axa, plot_name, 'eh? eh??')
    print(data)

    # Put mouse on columns
    data = data.unstack('mouse')

    continue

# print(zip(axa, plot_names))
#
# for plot_name in plot_names:
#     # Get data (corresponding column)
#     data = perf_metrics[plot_names]
#     print(data)