import pandas
import matplotlib
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

midx = pandas.MultiIndex.from_product(
    [[.25, .45, .65], [.0001, .01, .1]],
    names=['mean_interval', 'var_interval'])
sliced_by_both_v2 = acoustic_scored_by_fraction_correct.loc[:, midx]

# Drop sessions where non-final versions used
sliced_by_both_v2 = sliced_by_both_v2.dropna()

# Mean within mouse
meaned_by_mouse = sliced_by_both_v2.groupby('mouse').mean()

# Mean within mouse
meaned_by_mouse = sliced_by_both_v2.groupby('mouse').mean()

# Stacked to create series so that the avg fc can be graphed into bar plot
stacked = slice_by_both.stack()

# defining the variable "x"
x = meaned_by_mouse.columns.values.tolist()
xtick_labels = ['{}\n{}'.format(rate, var) for rate, var in x]

# Making the bar graph and adding tick labels for the x-axis (Code with no space between triplets)
# f, ax = plt.subplots()
# ax.bar((1,2,3,4,5,6,7,8,9),stacked)
# ax.set_xticks([1,2,3,4,5,6,7,8,9])
# ax.set_xticklabels(xtick_labels)

# # Making the bar graph and adding tick labels for the x-axis (code with spave in between triplets
f, ax = plt.subplots()
# Color borders of bars and color bars themselves, adding standard deviation if standard error use .sem
ax.bar((1,2,3,5,6,7,9,10,11),stacked, edgecolor = ['black'], color = ['white', 'grey', 'black'], yerr=meaned_by_mouse.std())
ax.set_xticks([1,2,3,5,6,7,9,10,11])
ax.set_xticklabels(xtick_labels)
# Making a legend (needs to complete)
f.legend(['High', 'Medium', 'Low'])
# Adding the circles with the preformace of each mouse
# ax.plot([1,2,3,5,6,7,9,10,11],meaned_by_mouse.T.values, marker='o', linestyle='none', markerfacecolor='none')

# create x and y labels
ax.set_ylabel('Fraction Correct')
ax.set_xlabel('Rate and Regularity')
## This line has to go at the end to make the plot show up (sometimes)
plt.show()

