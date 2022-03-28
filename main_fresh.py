import socket
import numpy as np
import pandas
import extras

# Check to see which computer it's running on and get the right file paths
computer = socket.gethostname()
print("Running on", computer)
if computer == 'squid':
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
    path_to_terminal_data = '/home/rowan/mnt/cuttlefish/behavior/from_octopus/autopilot/terminal/autopilot/data'
elif computer == 'clownfish':
    logfilenames = [
    ]
    path_to_terminal_data = 'x'
elif computer == 'octopus':
    path_to_terminal_data = 'y'
elif computer == 'x':  # Chris, put your office computer here
    path_to_terminal_data = 'z'
else:
    logfilenames = input("Computer not recognized. Please enter the filepath to the log files:")
    # TO DO: Figure out how to let user browse and select files instead of just inputting a filepath
    path_to_terminal_data= input("Please enter the filepath to the terminal data:")

# Specify data to load
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

# Checks mouse_names against cohorts and makes sure they have the same subjects. Appears to not allow duplicates either?
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
], names=['session_name', 'trial'])

# Load trial data and weights from the HDF5 files
# This also drops munged sessions
session_df, trial_data = extras.load_data_from_all_mouse_hdf5(
    mouse_names, munged_sessions, path_to_terminal_data,
)
# Load all logfiles into a huge list of lines
# Read each
all_logfile_lines = []
for logfilename in logfilenames:
    with open(logfilename) as fi:
        this_lines = fi.readlines()
    all_logfile_lines += this_lines

# with open("alllogfiles.txt", 'w') as output:
#     for row in all_logfile_lines:
#         output.write(str(row) + '\n')

# Calculate trial duration in seconds and add it as a column on trial_data
trial_data['duration'] = trial_data.groupby(
    ['mouse', 'session_name'])['timestamp'].diff() .shift(-1).apply(
    lambda dt: dt.total_seconds())
breakpoint()
# Parse the logfile lines
# These are all indexed by arbitrary 'n_session' instead of meaningful names
logfile_session_start_lines, logfile_trial_starts_df, logfile_pokes_df = (
    extras.parse_logfile_lines(all_logfile_lines)
    )

# This should be datetime64 not pandas.Timestamp
logfile_pokes_df['timestamp'] = logfile_pokes_df['timestamp'].apply(lambda x: x.to_datetime64())
print("The END")
