## Example of how to make plots from perf_metrics
# First, run main2_annotated, which will dump the following files:
#   perf_metrics
#   acoustic_scored_by_n_ports
#   acoustic_scored_by_fraction_correct
#   session_df
#
# This script demonstrates some examples of how to make plots from these data.

import pandas
import matplotlib.pyplot as plt


## Load the pickles
# session_df : Information about the sessions
#   index: MultiIndex, with levels (mouse, session_name)
#   columns: various metadata
#       'box', 'date', 'orig_session_num', 'first_trial', 'last_trial',
#       'n_trials', 'approx_duration', 'weight'
session_df = pandas.read_pickle(
    'session_df')
    
# perf_metrics : Overall performance for each session
#   index: MultiIndex, with levels (mouse, date)
#       This is the same as `session_df`, except that the session_name has
#       been replaced by the date of the session.
#   columns: various performance metrics
#       'session_name', 'rcp', 'fc', 'n_trials', 'ctrl_dur', 'opto_dur',
#       'weight'
perf_metrics = pandas.read_pickle(
    'perf_metrics')

# acoustic_scored_by_n_ports
#   index: MultiIndex, with levels (mouse, session_name)
#   columns: MultiIndex, with levels (mean_interval, var_interval)
#   values: The "rank of correct port" (rcp) for every combination of mouse,
#     session_name, mean_interval, and var_interval
acoustic_scored_by_n_ports = pandas.read_pickle(
    'acoustic_scored_by_n_ports')

# acoustic_scored_by_fraction_correct
#   Same as acoustic_scored_by_n_ports, except that the values are the
#   "fraction correct" instead of "rank of correct port"
acoustic_scored_by_fraction_correct = pandas.read_pickle(
    'acoustic_scored_by_fraction_correct')


## Plot the overall performance by day
# Get fraction correct on each date
overall_perf_by_day = perf_metrics['fc']

# Unstack mouse (ie put mouse on columns)
# Take a look at the difference between unstacked and overall_perf_by_day
# to see what unstacking does!
unstacked = overall_perf_by_day.unstack('mouse')

# Calculate the mean performance over mice
# Exercise: What is the difference between `axis=0` and `axis=1` here? Try it
mean_mouse = unstacked.mean(axis=1)

# Plot
f, ax = plt.subplots(figsize=(9, 3))
ax.plot(unstacked) # This plots each column (mouse) as a line
ax.set_ylabel('fraction correct')
ax.set_xlabel('date')

# Make a legend using the mouse names on the columns of `unstacked`
f.legend(unstacked.columns.values)

# Plot the mean mouse as a thick black dashed line
ax.plot(mean_mouse, lw=2, color='black', linestyle='--')

# Exercise: Make the same plot with rcp instead of fc


## Plot the performance by acoustic parameter
# First, mean over mice
# Look at this variable
mean_acoustic_score = acoustic_scored_by_fraction_correct.mean()

# Unstack the var_interval
# Look at this variable
# Why are there null (nan) values in it?
mean_acoustic_score_unstacked = mean_acoustic_score.unstack()

# Include only the final version of the parameters
# Compare these three ways of slicing ... What is the difference?
slice_by_rows = mean_acoustic_score_unstacked.loc[[.25, .45, .65], :]
slice_by_cols = mean_acoustic_score_unstacked.loc[:, [.0001, .01, .1]]
slice_by_both = mean_acoustic_score_unstacked.loc[
    [.25, .45, .65], [.0001, .01, .1]]

# Plot
# Question: Why is mean_interval on the x-axis and not var_interval?
f, ax = plt.subplots()
ax.plot(slice_by_both)
ax.set_xlabel('mean_interval')
ax.set_ylabel('fraction correct')

# Make a legend using the var_interval names on the columns of `slice_by_both`
f.legend(slice_by_both.columns.values)

# Exercise: Make the same plot, but with var_interval on the x-axis 
# instead of mean_interval
# Exercise: Make the same plot, but just for one mouse (e.g., F3_PAFT).
#   Hint: do this by slicing `acoustic_scored_by_fraction_correct`
#   before taking the mean.


## Bonus problem: What does this part do? Look at the variables.
# Slice columns with MultiIndex
midx = pandas.MultiIndex.from_product(
    [[.25, .45, .65], [.0001, .01, .1]], 
    names=['mean_interval', 'var_interval'])
sliced_by_both_v2 = acoustic_scored_by_fraction_correct.loc[:, midx]

# Drop sessions where non-final versions used
sliced_by_both_v2 = sliced_by_both_v2.dropna()

# Mean within mouse
meaned_by_mouse = sliced_by_both_v2.groupby('mouse').mean()


## This line has to go at the end to make the plot show up (sometimes)
plt.show()
