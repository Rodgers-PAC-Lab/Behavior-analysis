# Load PAFT data

"""
These are the essential backups to run every day
rsync -va --backup-dir=/home/chris/mnt/farscape_home/behavior/autopilot/logfiles/rpi01_backup_`date +%F_%H-%M-%S` /home/chris/mnt/rpi01/autopilot/logs/ /home/chris/mnt/farscape_home/behavior/autopilot/logfiles/rpi01
rsync -va --backup-dir=/home/chris/mnt/farscape_home/behavior/autopilot/logfiles/rpi05_backup_`date +%F_%H-%M-%S` /home/chris/mnt/rpi05/autopilot/logs/ /home/chris/mnt/farscape_home/behavior/autopilot/logfiles/rpi05
rsync -va --backup-dir=/home/chris/mnt/farscape_home/behavior/autopilot/terminal_backup_`date +%F_%H-%M-%S` /home/chris/autopilot /home/chris/mnt/farscape_home/behavior/autopilot/terminal
rsync -va /home/chris/Videos /mnt/farscape_x1/paft_videos/spinview
"""
import datetime
import importlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import extras
import my.plot
importlib.reload(extras)


## Specify data to load
# List of mouse names
# Session and trial data are loaded from the HDF5 files for each mouse
mouse_names = [
    'M1_PAFT', 'M3_PAFT',
    'M2_PAFT', 'M4_PAFT',
    'F1_PAFT', 'F3_PAFT',
    'F2_PAFT', 'F4_PAFT',
    ]

cohorts = {
    'all': [
        'M1_PAFT', 'M3_PAFT',
        'M2_PAFT', 'M4_PAFT',
        'F1_PAFT', 'F3_PAFT',
        'F2_PAFT', 'F4_PAFT',      
        ],
}
assert sorted(mouse_names) == sorted(np.concatenate(list(cohorts.values())))

# List of munged sessions
# These will be dropped

munged_sessions = [
    '20220302155704-M2_PAFT-Box2',
    '20220302160131-M2_PAFT-Box2',
    '20220302162326-M2_PAFT-Box2',
    '20220302162841-M4_PAFT-Box2',
    '20220307153240-F2_PAFT-Box2',
    '20220228172631-F1_PAFT-Box2',
    '20220321103222-M1_PAFT-Box2',
    '20220321104048-M3_PAFT-Box2',
]

# These are ones where there are no pokes, or no correct pokes
munged_trials = pandas.MultiIndex.from_tuples([
    ('20220302131754-F1_PAFT-Box2', 24), 
    ('20220303112827-F4_PAFT-Box2', 3),
    ('20220301141022-M3_PAFT-Box2', 6), 
    ('20220301145022-M2_PAFT-Box2', 11), 
    ('20220301152020-M4_PAFT-Box2', 8), 
    ('20220301161842-F4_PAFT-Box2', 3), 
    ('20220303153602-F3_PAFT-Box2', 9), 
    ('20220304112222-F3_PAFT-Box2', 21),
    ('20220307165025-M2_PAFT-Box2', 3), 
    ('20220307165025-M2_PAFT-Box2', 8),
    ('20220310133514-M2_PAFT-Box2', 18), 
    ('20220310140813-M4_PAFT-Box2', 11),
    ('20220311164333-M4_PAFT-Box2', 59),
    ('20220314104328-M3_PAFT-Box2', 9),
    ('20220316133933-F1_PAFT-Box2', 1),
    ('20220316140736-F3_PAFT-Box2', 22),
    ('20220316114550-M4_PAFT-Box2', 29),
    ('20220317125801-M2_PAFT-Box2', 48),
    ('20220317161245-F1_PAFT-Box2', 16), 
    ('20220318123209-F1_PAFT-Box2', 45), 
    ('20220318153029-F2_PAFT-Box2', 61),
    ('20220321101305-M1_PAFT-Box2', 43), 
    ('20220321113703-F1_PAFT-Box2', 13),
    ('20220322162928-M2_PAFT-Box2', 29), 
    ('20220322155709-M4_PAFT-Box2', 49),
    ('20220321142307-F4_PAFT-Box2', 61), 
    ('20220321171949-M4_PAFT-Box2', 35), 
    ('20220322140258-M3_PAFT-Box2', 14),
    ('20220323114916-F2_PAFT-Box2', 52),
    ('20220324151921-M1_PAFT-Box2', 5),
    ('20220325101431-M1_PAFT-Box2', 43)
    ], names=['session_name', 'trial'])

# List of logfilenames
# Poke times are loaded from these files
# Concatenate in this order
#~ logfilenames = [
    #~ '/home/chris/mnt/farscape_home/behavior/autopilot/logfiles/rpi01/tasks.log.5',
    #~ '/home/chris/mnt/farscape_home/behavior/autopilot/logfiles/rpi01/tasks.log.4',
    #~ '/home/chris/mnt/farscape_home/behavior/autopilot/logfiles/rpi01/tasks.log.3',
    #~ '/home/chris/mnt/farscape_home/behavior/autopilot/logfiles/rpi01/tasks.log.2',
    #~ '/home/chris/mnt/farscape_home/behavior/autopilot/logfiles/rpi01/tasks.log.1',
    #~ '/home/chris/mnt/farscape_home/behavior/autopilot/logfiles/rpi01/tasks.log',
    #~ '/home/chris/mnt/farscape_home/behavior/autopilot/logfiles/rpi05/tasks.log.3',
    #~ '/home/chris/mnt/farscape_home/behavior/autopilot/logfiles/rpi05/tasks.log.2',
    #~ '/home/chris/mnt/farscape_home/behavior/autopilot/logfiles/rpi05/tasks.log.1',
    #~ '/home/chris/mnt/farscape_home/behavior/autopilot/logfiles/rpi05/tasks.log',
    #~ ]
#~ logfilenames = [
    #~ '/home/chris/mnt/rpi01/autopilot/logs/tasks.log.9',
    #~ '/home/chris/mnt/rpi01/autopilot/logs/tasks.log.8',
    #~ '/home/chris/mnt/rpi01/autopilot/logs/tasks.log.7',
    #~ '/home/chris/mnt/rpi01/autopilot/logs/tasks.log.6',
    #~ '/home/chris/mnt/rpi01/autopilot/logs/tasks.log.5',
    #~ '/home/chris/mnt/rpi01/autopilot/logs/tasks.log.4',
    #~ '/home/chris/mnt/rpi01/autopilot/logs/tasks.log.3',
    #~ '/home/chris/mnt/rpi01/autopilot/logs/tasks.log.2',
    #~ '/home/chris/mnt/rpi01/autopilot/logs/tasks.log.1',
    #~ '/home/chris/mnt/rpi01/autopilot/logs/tasks.log',
    #~ '/home/chris/mnt/rpi05/autopilot/logs/tasks.log.7',
    #~ '/home/chris/mnt/rpi05/autopilot/logs/tasks.log.6',
    #~ '/home/chris/mnt/rpi05/autopilot/logs/tasks.log.5',
    #~ '/home/chris/mnt/rpi05/autopilot/logs/tasks.log.4',
    #~ '/home/chris/mnt/rpi05/autopilot/logs/tasks.log.3',
    #~ '/home/chris/mnt/rpi05/autopilot/logs/tasks.log.2',
    #~ '/home/chris/mnt/rpi05/autopilot/logs/tasks.log.1',
    #~ '/home/chris/mnt/rpi05/autopilot/logs/tasks.log',
    #~ ]
logfilenames = [
    '/home/rowan/mnt/cuttlefish/behavior/from_octopus/autopilot/logfiles/rpi05/tasks.log.8',
    '/home/rowan/mnt/cuttlefish/behavior/from_octopus/autopilot/logfiles/rpi05/tasks.log.7',
    '/home/rowan/mnt/cuttlefish/behavior/from_octopus/autopilot/logfiles/rpi05/tasks.log.6',
    '/home/rowan/mnt/cuttlefish/behavior/from_octopus/autopilot/logfiles/rpi05/tasks.log.5',
    '/home/rowan/mnt/cuttlefish/behavior/from_octopus/autopilot/logfiles/rpi05/tasks.log.4',
    '/home/rowan/mnt/cuttlefish/behavior/from_octopus/autopilot/logfiles/rpi05/tasks.log.3',
    '/home/rowan/mnt/cuttlefish/behavior/from_octopus/autopilot/logfiles/rpi05/tasks.log.2',
    '/home/rowan/mnt/cuttlefish/behavior/from_octopus/autopilot/logfiles/rpi05/tasks.log.1',
    '/home/rowan/mnt/cuttlefish/behavior/from_octopus/autopilot/logfiles/rpi05/tasks.log',
    ]


#~ # Glob logfilenames to see if we're missing any
#~ rpi01_logfiles = sorted(glob.glob(os.path.expanduser(
    #~ '~/mnt/rpi01/autopilot/logs/tasks.log*')))[::-1]
#~ rpi05_logfiles = sorted(glob.glob(os.path.expanduser(
    #~ '~/mnt/rpi05/autopilot/logs/tasks.log*')))[::-1]
#~ globbed_logfiles = rpi01_logfiles + rpi05_logfiles    

#~ # Check
#~ assert logfilenames == globbed_logfiles


## Load trial data and weights from the HDF5 files
# This also drops munged sessions
session_df, trial_data = extras.load_data_from_all_mouse_hdf5(
    mouse_names, munged_sessions,
    path_to_terminal_data='/home/rowan/mnt/cuttlefish/behavior/from_octopus/autopilot/terminal/autopilot/data',
    )
    

## Load all logfiles into a huge list of lines
# Read each
all_logfile_lines = []
for logfilename in logfilenames:
    with open(logfilename) as fi:
        this_lines = fi.readlines()
    all_logfile_lines += this_lines

# Calculate trial duration
trial_data['duration'] = trial_data.groupby(
    ['mouse', 'session_name'])['timestamp'].diff().shift(-1).apply(
    lambda dt: dt.total_seconds())


## Parse the logfile lines
# These are all indexed by arbitrary 'n_session' instead of meaningful names
logfile_session_start_lines, logfile_trial_starts_df, logfile_pokes_df = (
    extras.parse_logfile_lines(all_logfile_lines)
    )

# This should be datetime64 not pandas.Timestamp
logfile_pokes_df['timestamp'] = logfile_pokes_df['timestamp'].apply(lambda x: x.to_datetime64())


## Align logfile_trial_starts_df with the HDF5 stuff (session_df and trial_data)
logfile_hdf5_alignment = extras.align_logfile_with_hdf5(
    logfile_trial_starts_df, trial_data, session_df)

# Use the alignment to change the index on the logfile data to the more
# meaningful session names

def convert_logfile_index_to_hdf5_index(df, logfile_hdf5_alignment, 
    new_index, drop_n_session=True):
    # Relabel the stuff from the logfile with the canonical mouse/session labels
    df = df.join(logfile_hdf5_alignment, on='n_session')

    # Drop sessions from logfile for which we don't have HDF5 data
    df = df.dropna(subset=['mouse', 'session_name'])

    # Use mouse, session_name, and trial as index
    df = df.reset_index().set_index(new_index).sort_index()
    
    # Optionally drop the old index
    if drop_n_session:
        df = df.drop('n_session', axis=1)
    
    return df

# This changes the index and drops the unaligned data from the logfile
logfile_trial_starts_df = convert_logfile_index_to_hdf5_index(
    logfile_trial_starts_df, logfile_hdf5_alignment,
    new_index=['mouse', 'session_name', 'trial'])

logfile_pokes_df = convert_logfile_index_to_hdf5_index(
    logfile_pokes_df, logfile_hdf5_alignment,
    new_index=['mouse', 'session_name', 'poke'])

# Keep the n_session in this one for debugging
logfile_session_start_lines = convert_logfile_index_to_hdf5_index(
    logfile_session_start_lines, logfile_hdf5_alignment,
    new_index=['mouse', 'session_name'], drop_n_session=False)


## Check that logfile_trial_starts_df and trial_data match up
# There are often more trials in logfile_trial_starts_df and that's fine
n_trials1 = trial_data.groupby(
    ['mouse', 'session_name']).size().sort_index()
n_trials2 = logfile_trial_starts_df.groupby(
    ['mouse', 'session_name']).size().sort_index()
diff_trials = (n_trials2 - n_trials1).sort_values()

# diff_trials is always 0 or 1, except when something crashes and the
# HDF5 file stops storing data
assert diff_trials.drop([
    '20210920141403-Female3_0903-Box1',
    '20210920141407-Male3_0720-Box2',
    '20210907160408-Male1_0720-Box1',
    ], level='session_name', errors='ignore'
    ).isin([0, 1]).all()

# Compare equality of trial parameters
# Ignore trials that are missing from trial_data
sliced_ltsdf = logfile_trial_starts_df.loc[trial_data.index]

# These columns should be exactly equal
check_cols = ['rpi', 'side', 'sound', 'light', 'opto']
assert sliced_ltsdf[check_cols].equals(trial_data[check_cols])

# This can be slightly off, I guess due to logging latency
timestamp_diff = trial_data['timestamp'] - sliced_ltsdf['timestamp']
assert timestamp_diff.min() > datetime.timedelta(seconds=0)
assert timestamp_diff.max() < datetime.timedelta(seconds=.2)

# From now on, logfile_trial_starts_df is useless, except possibly for
# extracting the time of the last trial start, which is not in HDF5


## Assign pokes to each trial
# Error check same sessions contained in both
session_names = logfile_pokes_df.index.get_level_values(
    'session_name').unique().sort_values()
session_names_check = trial_data.index.get_level_values(
    'session_name').unique().sort_values()
assert session_names.equals(session_names_check)

# Align each session
poke_trials_l = []
poke_trials_keys_l = []
for (mouse, session_name), sub_trials in trial_data.groupby(['mouse', 'session_name']):
    # Get the pokes from this session
    sub_pokes = logfile_pokes_df.loc[mouse].loc[session_name].copy()
    
    # Drop these redundant levels
    sub_trials = sub_trials.droplevel('mouse').droplevel('session_name')
    
    # Sort both by timestamp
    sub_pokes = sub_pokes.sort_values('timestamp')
    sub_trials = sub_trials.sort_values('timestamp')
    
    # Check that trials are numbered starting from zero, so that searchsorted
    # can be interpreted as a trial number, rather than index into trial number
    assert (
        sub_trials.index.values == np.arange(len(sub_trials), dtype=int)).all()
    
    # Align
    # The -1 makes the pokes before the first trial timestamp to be -1
    # Can get errors here if anything is not datetime
    trial_of_each_poke = np.searchsorted(
        sub_trials['timestamp'].values, sub_pokes['timestamp'].values) - 1
    
    # Store the trial of each poke
    trial_of_each_poke_ser = pandas.Series(
        trial_of_each_poke, index=sub_pokes.index)
    
    # Store
    poke_trials_l.append(trial_of_each_poke_ser)
    poke_trials_keys_l.append((mouse, session_name))

# Concat over sessions
poke_trials_ser = pandas.concat(
    poke_trials_l, keys=poke_trials_keys_l, names=['mouse', 'session_name'])

# Insert this trial number into logfile_pokes_df
logfile_pokes_df['trial'] = poke_trials_ser
assert not logfile_pokes_df['trial'].isnull().any()

# Drop the pokes that happen before the first trial
# These are logged, but the correct port hasn't even been chosen yet
logfile_pokes_df = logfile_pokes_df[logfile_pokes_df['trial'] != -1].copy()

# Add trial to the index, between session_name and poke
logfile_pokes_df = logfile_pokes_df.set_index(
    'trial', append=True).swaplevel('poke', 'trial').sort_index()


## Fix the "L" and "R" pokes
fixed_pokes_l = []
for session_name, sub_pokes in logfile_pokes_df.groupby('session_name'):
    # Identify which box
    box = session_df.xs(session_name, level='session_name')['box'].item()

    # Replace
    if box == 'Box1':
        sub_pokes.loc[sub_pokes['port'] == 'L', 'port'] = 'rpi01_L'
        sub_pokes.loc[sub_pokes['port'] == 'R', 'port'] = 'rpi01_R'
    elif box == 'Box2':
        sub_pokes.loc[sub_pokes['port'] == 'L', 'port'] = 'rpi05_L'
        sub_pokes.loc[sub_pokes['port'] == 'R', 'port'] = 'rpi05_R'
    
    # Store
    fixed_pokes_l.append(sub_pokes)

# Concat
logfile_pokes_df = pandas.concat(fixed_pokes_l, verify_integrity=True)


## Add columns to trial_data, and calculate t_wrt_start for pokes
# Rename 'timestamp' to 'trial_start' for trials
trial_data = trial_data.rename(columns={'timestamp': 'trial_start'})

# Create "rpi_side" column
trial_data['rpi_side'] = trial_data['rpi'].str.cat(trial_data['side'], sep='_')

# Calculate t_wrt_start
logfile_pokes_df = logfile_pokes_df.join(trial_data['trial_start'])
logfile_pokes_df['t_wrt_start'] = (
    logfile_pokes_df['timestamp'] - logfile_pokes_df['trial_start'])
logfile_pokes_df['t_wrt_start'] = logfile_pokes_df['t_wrt_start'].apply(
    lambda ts: ts.total_seconds())

# Rename
logfile_pokes_df = logfile_pokes_df.rename(columns={'port': 'rpi_side'})


## Identify prev_target
prev_target_l = []
for keys, subtm in trial_data.groupby(['mouse', 'session_name']):
    prev_target = subtm['rpi_side'].shift()
    prev_target_l.append(prev_target)
trial_data['prev_rpi_side'] = pandas.concat(prev_target_l)


## Join target and prev_target on pokes, and define poke "typ2"
logfile_pokes_df = logfile_pokes_df.join(
    trial_data[['rpi_side', 'prev_rpi_side']], 
    rsuffix='_correct')

# target port is 'correct'
# prev_target port is 'prev' (presumably consumptive)
# all others (currently only one other type) is 'error'
logfile_pokes_df['typ2'] = 'error'
logfile_pokes_df.loc[
    logfile_pokes_df['rpi_side'] == logfile_pokes_df['rpi_side_correct'],
    'typ2'] = 'correct'
logfile_pokes_df.loc[
    logfile_pokes_df['rpi_side'] == logfile_pokes_df['prev_rpi_side'],
    'typ2'] = 'prev'

# Error check nothing weird happens when prev_target is null
assert logfile_pokes_df.loc[
    logfile_pokes_df['prev_rpi_side'].isnull(), 'typ2'].isin(
    ['error', 'correct']).all()


## Drop the broken trials
logfile_pokes_df = my.misc.slice_df_by_some_levels(
    logfile_pokes_df, munged_trials, drop=True)
trial_data = my.misc.slice_df_by_some_levels(
    trial_data, munged_trials, drop=True)


## Count pokes per trial and check there was at least 1 on every trial
# If error here, use trial_data['n_pokes'].isnull().sort_values()
# to find the munged trial and add to munged_trials
n_pokes = logfile_pokes_df.groupby(
    ['mouse', 'session_name', 'trial']).size().rename('n_pokes')
trial_data = trial_data.join(n_pokes)
#~ assert not trial_data['n_pokes'].isnull().any()
#~ assert (trial_data['n_pokes'] > 0).all()

# Find the files to add to the munged list
zero_pokes_trials = trial_data.index[trial_data['n_pokes'] == 0]
null_pokes_trials = trial_data.index[trial_data['n_pokes'].isnull()]

# I don't think it's possible for it to be zero
assert len(zero_pokes_trials) == 0

# Suggest trials to add to munged trials
suggestion = list(null_pokes_trials.to_frame()[
    ['session_name', 'trial']].reset_index(drop=True).to_records(index=False))
if len(suggestion) > 0:
    print("error: some trials have no pokes. Add this to munged_trials:\n{}".format(suggestion))


## Debug there is always a correct poke
# For some reason there just isn't a correct poke on a few trials here and there
n_poke_types_by_trial = logfile_pokes_df.groupby(
    ['session_name', 'trial'])['typ2'].value_counts().unstack(
    'typ2').fillna(0).astype(int)
assert len(trial_data) == len(n_poke_types_by_trial)
#~ assert (n_poke_types_by_trial['correct'] > 0).all()

# Suggest trials to add to munged trials
no_correct_pokes_trials = n_poke_types_by_trial.index[
    n_poke_types_by_trial['correct'] == 0]
suggestion = list(no_correct_pokes_trials.to_frame()[
    ['session_name', 'trial']].reset_index(drop=True).to_records(index=False))
if len(suggestion) > 0:
    print("error: some trials have no correct pokes. Add this to munged_trials:\n{}".format(suggestion))


## Score by n_trials
scored_by_n_trials = trial_data.groupby(['mouse', 'session_name']).size()


## Score the trials by correct
# time of first poke of each type
first_poke = logfile_pokes_df.reset_index().groupby(
    ['mouse', 'session_name', 'trial', 'typ2']
    )['t_wrt_start'].min().unstack('typ2')

# Join
trial_data = trial_data.join(first_poke)

# Score by first poke (excluding prev)
trial_outcome = trial_data[['correct', 'error']].idxmin(1)
trial_data['outcome'] = trial_outcome

# Score
scored = trial_data.groupby(
    ['mouse', 'session_name', 'sound', 'light'])[
    'outcome'].value_counts().unstack('outcome')
scored['perf'] = scored['correct'].divide(scored['correct'] + scored['error'])

# unstack stim type
scored_by_fraction_correct = scored['perf'].unstack(['sound', 'light']).sort_index(axis=1)

# Same for opto instead of now-useless sound and light
opto_scored = trial_data.groupby(
    ['mouse', 'session_name', 'opto'])[
    'outcome'].value_counts().unstack('outcome')
opto_scored['perf'] = opto_scored['correct'].divide(opto_scored['correct'] + opto_scored['error'])
opto_scored_by_fraction_correct = opto_scored['perf'].unstack('opto').sort_index(axis=1)
opto_scored_by_fraction_correct = opto_scored_by_fraction_correct.rename(
    columns={False: 'ctrl', True: 'opto'})

# Score by acoustic
joined = trial_data.join(logfile_trial_starts_df[['mean_interval', 'var_interval']])
acoustic_scored = joined.groupby(
    ['mouse', 'session_name', 'mean_interval', 'var_interval']
    )['outcome'].value_counts().unstack('outcome').fillna(0)
acoustic_scored['perf'] = acoustic_scored['correct'].divide(
    acoustic_scored['correct'] + acoustic_scored['error'])
acoustic_scored_by_fraction_correct = acoustic_scored['perf'].unstack(
    ['mean_interval', 'var_interval']).sort_index(axis=1)


## Score trials by how many ports poked before correct
# Get the latency to each port on each trial
latency_by_port = logfile_pokes_df.reset_index().groupby(
    ['mouse', 'session_name', 'trial', 'rpi_side'])['t_wrt_start'].min()

# Drop the consumption port (previous reward)
consumption_port = trial_data[
    'prev_rpi_side'].dropna().reset_index().rename(
    columns={'prev_rpi_side': 'rpi_side'})
cp_idx = pandas.MultiIndex.from_frame(consumption_port)
latency_by_port_dropped = latency_by_port.drop(cp_idx, errors='ignore')

# Unstack the port onto columns
lbpd_unstacked = latency_by_port_dropped.unstack('rpi_side')

# Rank them in order of poking
# Subtract 1 because it starts with 1
# The best is 0 (correct trial) and the worst is 6 (because consumption port
# is ignored). The expectation under random choices is 3 (right??)
lbpd_ranked = lbpd_unstacked.rank(
    method='first', axis=1).stack().astype(int) - 1

# Find the rank of the correct port
correct_port = trial_data[
    'rpi_side'].dropna().reset_index()
cp_idx = pandas.MultiIndex.from_frame(correct_port)
rank_of_correct_port = lbpd_ranked.reindex(
    cp_idx).droplevel('rpi_side').rename('rcp')

# Append this to big_trial_data
trial_data = trial_data.join(rank_of_correct_port)

# Error check
assert not trial_data['rcp'].isnull().any()


## Count errors by position wrt goal
# Define port_angle. Compass convention: 0 = North, increases CW
port2angle_box1 = pandas.Series(
    [135, 180, 225, 270, 315, 0, 45, 90],
    index=[
    'rpi01_L', 'rpi01_R', 'rpi02_L', 'rpi02_R',
    'rpi03_L', 'rpi03_R', 'rpi04_L', 'rpi04_R',
    ])
port2angle_box2 = pandas.Series(
    [135, 180, 225, 270, 315, 0, 45, 90],
    index=[
    'rpi05_L', 'rpi05_R', 'rpi06_L', 'rpi06_R',
    'rpi07_L', 'rpi07_R', 'rpi08_L', 'rpi08_R',
    ])
port2angle = pandas.concat([port2angle_box1, port2angle_box2],  
    keys=['Box1', 'Box2'], names=['box'])

# This section reuses lbpd_unstacked and correct_port from above
# First just identify which ports were poked at all
ports_poked_by_trial = ~lbpd_unstacked.isnull()

# This is the same as ports_poked_by_trial, but is True only for first port
first_port_poked_by_trial = lbpd_unstacked.le(lbpd_unstacked.min(1), axis=0)
ports_poked_by_trial = first_port_poked_by_trial.copy()

# Identify box of each trial
session2box = session_df['box'].droplevel('mouse')
box_level = ports_poked_by_trial.index.get_level_values(
    'session_name').map(session2box).rename('box')
new_index = ports_poked_by_trial.index.to_frame().reset_index(drop=True)
new_index['box'] = box_level
ports_poked_by_trial.index = pandas.MultiIndex.from_frame(new_index)
ports_poked_by_trial = ports_poked_by_trial.reorder_levels(
    ['box', 'mouse', 'session_name', 'trial']).sort_index()

# Rename the ports into angular location for each box
#~ ppbt_box1 = ports_poked_by_trial.loc['Box1', 
    #~ port2angle.loc['Box1'].index.values]
#~ ppbt_box1.columns = port2angle.loc['Box1'].values
#~ ppbt_box1.columns.name = 'port_dir'

ppbt_box2 = ports_poked_by_trial.loc['Box2', 
    port2angle.loc['Box2'].index.values]
ppbt_box2.columns = port2angle.loc['Box2'].values
ppbt_box2.columns.name = 'port_dir'

# Collapse over boxes
ports_poked_by_trial = pandas.concat(
    [ppbt_box2], #[ppbt_box1, ppbt_box2], 
    keys=['Box2'], #keys=['Box1', 'Box2'], 
    names=['box']
    ).reorder_levels(['mouse', 'box', 'session_name', 'trial']
    ).sort_index().sort_index(axis=1)

# Calculate correct_dir and persev_dir for each trial
correct_dir = trial_data['rpi_side'].map(
    port2angle.droplevel('box')).rename('goal_dir')
perseverative_dir = trial_data['prev_rpi_side'].map(
    port2angle.droplevel('box')).rename('persev_dir')
assert not correct_dir.isnull().any()

# Reshape and keep only pokes (where poked == True)
unleveled = ports_poked_by_trial.stack().rename('poked').reset_index()
unleveled = unleveled.loc[unleveled['poked'] == True]

# Normalize the direction of each poke to the goal_dir
unleveled = unleveled.join(correct_dir, on=['mouse', 'session_name', 'trial'])
unleveled = unleveled.join(perseverative_dir, on=['mouse', 'session_name', 'trial'])
unleveled['err_dir'] = unleveled['port_dir'] - unleveled['goal_dir']
unleveled['persev_change'] = unleveled['port_dir'] - unleveled['persev_dir']

# Mod by 360 degrees, into the [0, 360) range
unleveled['err_dir'] = np.mod(unleveled['err_dir'], 360)
unleveled['persev_change'] = np.mod(unleveled['persev_change'], 360)

# Equivalently code into [-180, 180) range
unleveled.loc[unleveled['err_dir'] >= 180, 'err_dir'] -= 360
unleveled.loc[unleveled['persev_change'] >= 180, 'persev_change'] -= 360
unleveled.loc[unleveled['goal_dir'] >= 180, 'goal_dir'] -= 360
unleveled.loc[unleveled['port_dir'] >= 180, 'port_dir'] -= 360

# Remove temporary columns and reindex
err_dir_by_trial = unleveled.set_index(
    ['mouse', 'box', 'session_name', 'trial'])[
    ['err_dir', 'port_dir', 'persev_change']].sort_index()


## Score by n_ports
scored_by_n_ports = trial_data.groupby(
    ['mouse', 'session_name', 'sound', 'light'])['rcp'].mean().unstack(
    ['sound', 'light']).sort_index(axis=1)

# Same for opto instead of now-useless sound and light
opto_scored_by_n_ports = trial_data.groupby(
    ['mouse', 'session_name', 'opto'])['rcp'].mean().unstack(
    ['opto']).sort_index(axis=1)
opto_scored_by_n_ports = opto_scored_by_n_ports.rename(
    columns={False: 'ctrl', True: 'opto'})

# Same for mean_interval and var_interval
joined = trial_data.join(logfile_trial_starts_df[['mean_interval', 'var_interval']])
acoustic_scored_by_n_ports = joined.groupby(
    ['mouse', 'session_name', 'mean_interval', 'var_interval'])['rcp'].mean().unstack(
    ['mean_interval', 'var_interval']).sort_index(axis=1)


## Trial duration
trial_duration_by_opto = trial_data.groupby(
    ['mouse', 'session_name', 'opto'])['duration'].median().unstack('opto')
trial_duration_by_opto = trial_duration_by_opto.reset_index().set_index('session_name')
trial_duration_by_opto = trial_duration_by_opto.rename(
    columns={False: 'ctrl_dur', True: 'opto_dur'})


## Extract key performance metrics
# This slices out sound-only trials
perf_metrics = pandas.concat([
    scored_by_n_ports.loc[:, (True, False)].rename('rcp'),
    scored_by_fraction_correct.loc[:, (True, False)].rename('fc'),
    opto_scored_by_n_ports.loc[:, 'opto'].rename('opto_rcp'),
    opto_scored_by_n_ports.loc[:, 'ctrl'].rename('ctrl_rcp'),
    opto_scored_by_fraction_correct.loc[:, 'opto'].rename('opto_fc'),
    opto_scored_by_fraction_correct.loc[:, 'ctrl'].rename('ctrl_fc'),
    scored_by_n_trials.rename('n_trials'),
    ], axis=1, verify_integrity=True)

# Join trial_duration
perf_metrics = perf_metrics.join(
    trial_duration_by_opto[['ctrl_dur', 'opto_dur']], on='session_name')

# Join on weight and date
perf_metrics = perf_metrics.join(session_df[['date', 'weight']])

# Index by date
perf_metrics = perf_metrics.reset_index().set_index(
    ['mouse', 'date']).sort_index()


## Dump
perf_metrics.to_pickle('perf_metrics')


## Plots
# Choose dates to plot
all_dates = sorted(perf_metrics.index.levels[1])

# Choose recent dates (for perf plot), and the mice that are in them
recent_dates = all_dates[-18:]
recent_sessions = session_df.loc[
    session_df['date'].isin(recent_dates)]['date'].reset_index()
recent_mouse_names = sorted(recent_sessions['mouse'].unique())

# Choose very recent dates (for error plot), and the mice that are in them
very_recent_dates = recent_dates[-5:]
very_recent_sessions = session_df.loc[
    session_df['date'].isin(very_recent_dates)]['date'].reset_index()
very_recent_mouse_names = sorted(very_recent_sessions['mouse'].unique())

# Generate color bar
np.random.seed(0) # if you don't like it, change it
universal_colorbar = np.random.permutation(my.plot.generate_colorbar(len(recent_mouse_names)))
universal_colorbar = pandas.DataFrame(
    universal_colorbar, index=recent_mouse_names)


## Iterate over cohorts
for cohort in list(cohorts.keys()):# + ['all']:
    
    ## The overall plot for this cohort
    # Get mice in this cohort
    if cohort == 'all':
        # If all, use the recent ones (not the very recent ones)
        cohort_mice = recent_mouse_names
    else:
        cohort_mice = cohorts[cohort]

    # Plots
    plot_names = ['fc', 'rcp', 'n_trials', 'weight']
    f, axa = plt.subplots(len(plot_names), 1, figsize=(5, 8), sharex=False)
    f.subplots_adjust(hspace=.9, bottom=.1, top=.95)
    for ax, plot_name in zip(axa, plot_names):
        # Get data (corresponding column)
        data = perf_metrics[plot_name]
        
        # Put mouse on columns
        data = data.unstack('mouse')
        
        # Slice out just the cohort mice
        #~ data = data.loc[:, cohort_mice]
        data = data.reindex(cohort_mice, axis=1)
        
        # Slice out just the recent data
        data = data.loc[recent_dates]
        
        # Remove dates from index
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
            ax.plot(ax.get_xlim(), [1/7., 1/7.], 'k--', lw=.75)
        elif plot_name == 'rcp':
            ax.set_ylim((4, 0))
            ax.plot(ax.get_xlim(), [3., 3.], 'k--', lw=.75)
        elif plot_name == 'n_trials':
            ax.set_ylim(ymin=0)
        
        # Legend
        if plot_name == 'weight':
            ax.legend(fontsize=10, loc='lower left')
        
        xt = np.array(list(range(len(recent_dates))))
        xtl = [dt.strftime('%m-%d') for dt in recent_dates]
        ax.set_xticks(xt)
        ax.set_xticklabels(xtl, rotation=45)
        ax.set_xlim((xt[0] - .5, xt[-1] + .5))    
    continue

    ## Plot error distr over time
    # Plot each
    metrics_l = ['err_dir', 'port_dir', 'persev_change', 'legend']
    f, axa = plt.subplots(
        len(metrics_l), len(very_recent_dates), sharex=True, sharey=True,
        figsize=(14, 6))
    f.subplots_adjust(hspace=.3, wspace=.3, bottom=.1, top=.95, left=.06, right=.97)

    xticks = sorted(err_dir_by_trial['port_dir'].unique())
    for metric in metrics_l:
        for very_recent_date in very_recent_dates:
            # Get ax
            ax = axa[
                metrics_l.index(metric),
                very_recent_dates.index(very_recent_date),
                ]
            
            plotted_mouse_names_l = []
            for mouse_name in cohort_mice:
                # Get color by mouse
                try:
                    color = universal_colorbar.loc[mouse_name].values
                except KeyError:
                    color = 'k'
                
                # Get corresponding session
                try:
                    session_name = very_recent_sessions.loc[
                        (very_recent_sessions['date'] == very_recent_date) &
                        (very_recent_sessions['mouse'] == mouse_name),
                        'session_name'].item()
                except ValueError:
                    continue
                plotted_mouse_names_l.append(mouse_name)
                
                # Slice
                this_session = err_dir_by_trial.xs(
                    session_name, level='session_name').droplevel(['mouse', 'box'])
                
                # This is just to display a legend
                if metric == 'legend':
                    ax.plot([], [], color=color)

                else:
                    err_distr = this_session[metric].value_counts().sort_index()
                    err_distr = err_distr.reindex(xticks)
                    
                    # Normalize
                    n_trials = len(trial_data.xs(session_name, level='session_name'))
                    err_distr = err_distr / n_trials
                    
                    # Plot
                    ax.plot(err_distr, '.-', color=color)

            # Pretty ax
            if metric == 'legend':
                ax.legend(plotted_mouse_names_l, fontsize=5, loc='upper left')
            if ax in axa[:, 0]:
                ax.set_ylabel(metric)
            if ax in axa[0, :]:
                ax.set_title(very_recent_date)
            my.plot.despine(ax)

    ax.set_xlim((-180, 180))
    ax.set_xticks((-180, 0, 180))
    ax.set_ylim((0, 1))

plt.show()

## Slice acoustic by day
acoustic_scored_by_n_ports = acoustic_scored_by_n_ports.stack().stack().rename(
    'perf').reset_index().join(session_df['date'], on=[
    'mouse', 'session_name']).set_index(
    ['date', 'mouse', 'session_name', 'mean_interval', 'var_interval'])['perf'].unstack(
    ['mean_interval', 'var_interval'])

acoustic_scored_by_fraction_correct = acoustic_scored_by_fraction_correct.stack().stack().rename(
    'perf').reset_index().join(session_df['date'], on=[
    'mouse', 'session_name']).set_index(
    ['date', 'mouse', 'session_name', 'mean_interval', 'var_interval'])['perf'].unstack(
    ['mean_interval', 'var_interval'])


# Include only this day

acoustic_scored_by_n_ports = acoustic_scored_by_n_ports.loc[
    datetime.date(2022, 3, 23)].dropna(1)
acoustic_scored_by_fraction_correct = acoustic_scored_by_fraction_correct.loc[
    datetime.date(2022, 3, 23)].dropna(1)

#~ # Drop only this day
#~ acoustic_scored_by_n_ports = acoustic_scored_by_n_ports.drop(
    #~ datetime.date(2022, 3, 17)).dropna(1)
#~ acoustic_scored_by_fraction_correct = acoustic_scored_by_fraction_correct.drop(
    #~ datetime.date(2022, 3, 17)).dropna(1)


## Plot acoustic
metric_l = ['rcp', 'fc']
plot_by_l = ['mean_interval', 'var_interval']

f, axa = plt.subplots(2, 2)
f.subplots_adjust(hspace=.4, wspace=.4)
for metric in metric_l:
    for plot_by in plot_by_l:
        ax = axa[
            metric_l.index(metric),
            plot_by_l.index(plot_by),
            ]
        
        # Get data
        if metric == 'rcp':
            data = acoustic_scored_by_n_ports
        elif metric == 'fc':
            data = acoustic_scored_by_fraction_correct

        # Mean over mice and session
        data = data.mean()
        
        # Unstack plot_by
        data = data.unstack(plot_by)
        
        ax.plot(data, marker='o', linestyle='-')
        
        if plot_by == 'mean_interval':
            ax.set_xlabel('irregularity')
            ax.legend(['high rate', 'med rate', 'low rate'], loc='upper right', fontsize='x-small')
            
        elif plot_by == 'var_interval':
            ax.set_xlabel('inter-sound interval (s)')
            ax.legend(['regular', 'med', 'irregular'], loc='upper right', fontsize='x-small')
        
        if metric == 'rcp':
            ax.set_ylabel('rank of correct port')
            ax.set_ylim((3, 0))
            ax.set_yticks((0, 1, 2,3))
        elif metric == 'fc':
            ax.set_ylabel('fraction correct')
            ax.set_ylim((0, 1))
            ax.set_yticks((0, .5, 1))

## Show
plt.show()


## This is for examining data from today
todays_perf = perf_metrics.xs(datetime.date.today(), level='date').copy()
todays_perf['cohort'] = ''
for cohort, cohort_mice in cohorts.items():
    for mouse in cohort_mice:
        if mouse in todays_perf.index:
            todays_perf.loc[mouse, 'cohort'] = cohort

# This is for counting trials from today across all ports
n_trials_today_by_port = trial_data.reindex(
    todays_perf['session_name'].values, level='session_name').groupby(
    'rpi_side').size()

# This is for counting trials over the last three days
recent_n_trials_by_mouse = perf_metrics['n_trials'].unstack(
    'mouse').loc[very_recent_dates].dropna(axis=1, how='all').mean().sort_values()

# This is for looking at todays_perf
todays_perf = perf_metrics.xs(datetime.date.today(), level=1).drop(
    'session_name', axis=1).sort_index(axis=1)

