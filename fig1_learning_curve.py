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

#OH&MG mice (takes out the columns (mice)
#sliced unstacked data frame by columns.
slice_by_cols = unstacked.loc[:, ['F1_PAFT', 'F3_PAFT', 'M1_PAFT', 'M3_PAFT']]

# Calculate the mean performance over mice
# Exercise: What is the difference between `axis=0` and `axis=1` here? Try it
mean_mouse = slice_by_cols.mean(axis=1)

# Plot (of all mice) (commented out, new plot for slice_by_cols)
# f, ax = plt.subplots(figsize=(9, 3))
# ax.plot(unstacked.iloc[:15]) # This plots each column (mouse) as a line
# ax.set_ylabel('fraction correct')
# ax.set_xlabel('date')

# Plot (slice_by_cols i.e OH&MZ mice)
f, ax = plt.subplots(figsize=(9, 3))
ax.plot(slice_by_cols.iloc[:15]) # This plots each column (mouse) as a line
ax.set_title('Fraction Correct Over Time - OH&MZ')
ax.set_ylabel('Fraction Correct')
ax.set_xlabel('Time')
ax.set_ylim((0, 1))

# Make a legend using the mouse names on the columns of `unstacked`
f.legend(slice_by_cols.columns.values)

# Plot the mean mouse as a dotted black line
ax.plot(mean_mouse.iloc[:15], lw=2, color='black', linestyle='dotted')

# Making a highlighted section on the graph to indicate a break
ax.axvspan('2022-03-04', '2022-03-14', alpha=0.1, color='magenta')

#Plot the chance preformance line
ax.plot(ax.get_xlim(), [1 / 7., 1 / 7.], 'k--', lw=.75)


## This line has to go at the end to make the plot show up (sometimes)
plt.show()

#Plot after change in acoustic parameter for OH&MZ cohort
f, ax = plt.subplots(figsize=(7, 3))
ax.plot(slice_by_cols.iloc[15:]) # This plots each column (mouse) as a line
ax.set_title('Fraction Correct Over Time After Change in Acoustic Parameters - OH&MZ')
ax.set_ylabel('Fraction Correct')
ax.set_xlabel('Time')
ax.set_ylim((0, 1))

# Make a legend using the mouse names on the columns of `unstacked`
f.legend(slice_by_cols.columns.values)

# Plot the mean mouse as a dotted black line
ax.plot(mean_mouse.iloc[15:], lw=2, color='black', linestyle='dotted')

#Plot the chance preformance line
ax.plot(ax.get_xlim(), [1 / 7., 1 / 7.], 'k--', lw=.75)


#Plot for RG cohort
#sliced unstacked data frame by columns.
slice_by_cols_RG = unstacked.loc[:, ['F2_PAFT', 'F4_PAFT', 'M2_PAFT', 'M4_PAFT']]

# Calculate the mean performance over mice
# Exercise: What is the difference between `axis=0` and `axis=1` here? Try it
mean_mouse = slice_by_cols_RG.mean(axis=1)

# Plot (slice_by_cols_RG i.e RG mice)
f, ax = plt.subplots(figsize=(9, 3))
ax.plot(slice_by_cols_RG.iloc[:15]) # This plots each column (mouse) as a line
ax.set_title('Fraction Correct Over Time - RG')
ax.set_ylabel('Fraction Correct')
ax.set_xlabel('Time')
ax.set_ylim((0, 1))

# Make a legend using the mouse names on the columns of `unstacked`
f.legend(slice_by_cols_RG.columns.values)

# Plot the mean mouse as a dotted black line
ax.plot(mean_mouse.iloc[:15], lw=2, color='black', linestyle='dotted')

#Plot the chance preformance line
ax.plot(ax.get_xlim(), [1 / 7., 1 / 7.], 'k--', lw=.75)


## This line has to go at the end to make the plot show up (sometimes)
plt.show()

#Plot after change in acoustic parameter for OH&MG cohort
f, ax = plt.subplots(figsize=(7, 3))
ax.plot(slice_by_cols_RG.iloc[15:]) # This plots each column (mouse) as a line
ax.set_title('Fraction Correct Over Time After Change in Acoustic Parameters - RG')
ax.set_ylabel('Fraction Correct')
ax.set_xlabel('Time')
ax.set_ylim((0, 1))

# Make a legend using the mouse names on the columns of `unstacked`
f.legend(slice_by_cols_RG.columns.values)

# Plot the mean mouse as a dotted black line
ax.plot(mean_mouse.iloc[15:], lw=2, color='black', linestyle='dotted')

#Plot the chance preformance line
ax.plot(ax.get_xlim(), [1 / 7., 1 / 7.], 'k--', lw=.75)
