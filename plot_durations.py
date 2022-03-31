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
midx = pandas.MultiIndex.from_product(
    [[.25, .45, .65], [.0001, .01, .1]],
    names=['mean_interval', 'var_interval'])

simple_acoustic = acoustic_trials.drop(['light','sound','opto','trial_start','prev_rpi_side','n_pokes'],axis=1)
simple_acoustic = simple_acoustic.reset_index()
#simple_acoustic[["var_interval"]] = simple_acoustic[["var_interval"]].astype(float)
#simple_acoustic[["mean_interval"]] = simple_acoustic[["mean_interval"]].astype(float)
simple_acoustic['condition'] = simple_acoustic['mean_interval'].astype(str) + '/' + simple_acoustic['var_interval'].astype(str)
durations = simple_acoustic.groupby(['mouse','mean_interval','var_interval'])['duration'].mean().unstack('mouse')
clean_durations=durations.dropna()
# data = {
#     'mean_interval': durations,
#     'duration': durations['mouse']

#}
f, ax = plt.subplots(figsize=(9, 3))
ax.plot(clean_durations.unstack('var_interval'))
ax.set_ylabel('duration')
ax.set_xlabel('mean_interval')
f.legend(clean_durations.columns.values)
mean_mouse=clean_durations.unstack('var_interval').mean(axis=1)
ax.plot(mean_mouse, lw=2, color='black', linestyle='--')




avgby_var = simple_acoustic.groupby(['mean_interval','var_interval'])['duration'].mean().unstack('var_interval')
avgby_var=avgby_var.drop([0.0018,0.0320],axis=1)
avgby_var=avgby_var.drop([0.15,0.35,0.55],axis=0)
f, ax = plt.subplots(figsize=(9, 3))
ax.plot(avgby_var)
ax.set_ylabel('duration')
ax.set_xlabel('mean_interval')
f.legend(avgby_var.columns.values)


print("the end")