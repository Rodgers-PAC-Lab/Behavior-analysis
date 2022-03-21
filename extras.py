import os
import tables
import pandas
import numpy as np
import datetime
import scipy.optimize

def align_logfile_with_hdf5(logfile_trial_starts_df, trial_data, session_df):
    """Match up the logfile sessions with the HDF5 sessions.
    
    This extracts the time of the first trial of each session in the logfile,
    the time of the first trial of each session in the HDF5 file, and then
    uses the Hungarian algorithm to match them up with each other.
    
    Every entry in session_df will be aligned with an entry in 
    `logfile_trial_starts_df`, but the reverse is not true. Many entries in
    `logfile_trial_starts_df` correspond to sessions that should be ignored
    (such as tstPAFT sessions).
    
    Arguments:
        logfile_trial_starts_df : from parse_logfile_lines
        trial_data : from load_data_from_all_mouse_hdf5
        session_df : from load_data_from_all_mouse_hdf5
    
    Returns: DataFrame
        index : 'n_session', the arbitrary session numbers on the first level
            of the logfile_trial_starts_df index
        columns : 'mouse' and 'session_name'
            The levels of the session_df MultiIndex that align with that
            n_session
    """
    # Ensure we started parsing the logfile soon enough
    # If not, change `start_datetime` in identify_session_starts
    assert logfile_trial_starts_df['timestamp'].min() <= trial_data['timestamp'].min()

    # Get the start of each session, according to the logfiles
    logfile_session_start = logfile_trial_starts_df['timestamp'].groupby(
        'n_session').min().sort_values()

    # Get the start of each session, according to the HDF5 files
    hdf5_session_start = session_df['first_trial']

    # Align
    alignment_df = pandas.concat([logfile_session_start] * len(hdf5_session_start), 
        axis=1)
    alignment_df.columns = hdf5_session_start.index
    alignment_df = alignment_df.sub(hdf5_session_start)
    alignment_df = alignment_df.apply(lambda xxx: xxx.dt.total_seconds())
    alignment_df = alignment_df.abs()

    # Hungarian
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(alignment_df)
    aligned_l = []
    for rr, cc in zip(row_ind, col_ind):
        cost = alignment_df.iloc[rr, cc]
        bsd_index = alignment_df.index[rr]
        mouse, btd_session = alignment_df.columns[cc]
        aligned_l.append((bsd_index, mouse, btd_session, cost))
    aligned_df = pandas.DataFrame.from_records(
        aligned_l, columns=['n_session', 'mouse', 'session_name', 'cost'])

    # None should be that off
    assert aligned_df['cost'].max() < .1

    # Create a conversion DataFrame to relate arbitrary session numbers in the 
    # logfile to meaningful mouse and session names
    conversion = aligned_df.set_index('n_session')[['mouse', 'session_name']]
    assert len(conversion) == len(session_df)
    
    return conversion

def identify_session_starts(all_logfile_lines, 
    start_datetime=datetime.datetime(2021, 7, 20)):
    """Identify the line on which each session starts
    
    This searches for lines that look like this:
        '2021-06-10 12:58:37,758 - tasks.paft - INFO : parent, module-level logger created: tasks.paft\n'
    
    These lines are defined as the session start.
    
    Returns : session_starts, a DataFrame
        index : arbitrary integers called 'n_session'
        columns :
            'start_line' : the line number on which that session starts
            'stop_line' : the line number on which that session stops
            'dt_start' : datetime, the time of the session start
    """
    ## Identify session starts
    session_starts = []
    for nline, line in enumerate(all_logfile_lines):
        # Look for this line which indicates the first line in a session
        if 'INFO : parent, module-level logger created: tasks.' in line:
            # Extract the time
            spline = line.split()
            task_name = line.split()[3].replace('tasks.', '')
            date_string = spline[0]
            time_string = spline[1]
            session_start = datetime.datetime.strptime(
                date_string + ' ' + time_string, "%Y-%m-%d %H:%M:%S,%f")
            session_starts.append((nline, session_start, task_name))

    # DataFrame it
    session_starts = pandas.DataFrame.from_records(
        session_starts, columns=['start_line', 'dt_start', 'task'])

    # Assign stop line
    session_starts['stop_line'] = session_starts['start_line'].shift(-1)
    session_starts.loc[
        session_starts.index[-1], 'stop_line'] = len(all_logfile_lines)
    session_starts['stop_line'] = session_starts['stop_line'].astype(int)

    # Start in the recent past
    session_starts = session_starts.loc[
        session_starts['dt_start'] >= start_datetime]
    session_starts.index = range(len(session_starts))
    session_starts.index.name = 'n_session'
    
    return session_starts

def parse_lines_from_single_session(session_lines):
    """Parse logfile lines from a single session and identify poke times.

    The lines come in these flavors:
    This indicates the beginning of the session
    '2021-06-10 12:58:37,758 - tasks.paft - INFO : parent, module-level logger created: tasks.paft\n'

    This indicates the beginning of the trial
     '2021-06-10 13:41:08,888 - tasks.paft.PAFT - DEBUG : The chosen target is L\n',

    This indicates a poke
     '2021-06-10 13:02:09,098 - tasks.paft.PAFT - DEBUG : 2021-06-10T13:02:09.098430 L poke\n',
     '2021-06-10 13:41:09,238 - tasks.paft.PAFT - DEBUG : 2021-06-10T13:41:09.238338 rpi03_R poke\n',

    This indicates a correct poke (end of trial), except rpi01 doesn't generate these:
     "2021-06-10 13:03:31,075 - tasks.paft.PAFT - DEBUG : correct poke {'from': 'rpi04', 'poke': 'R'}; target was rpi04_R\n",

    This indicates an incorrect poke, except rpi01 doesn't generate these:
     "2021-06-12 14:46:56,963 - tasks.paft.PAFT - DEBUG : incorrect poke {'from': 'rpi03', 'poke': 'R'}; target was rpi04_L\n",

    
    Returns : pokes_df, starts_df
        These are all None if an error was encountered. Otherwise:
        pokes_df : DataFrame with length equal to number of pokes
            columns : 
                timestamp : datetime, timestamp of porty entry
                port : string, the port name (directly from logfile)
            
        starts_df : DataFrame with length equal to number of trials
            columns :
                timestamp : datetime, timestamp of trial start
                rpi : string, the target rpi
                side : string, the target side
                sound, light : bool
    """
    # Ignore lines with these tokens
    ignore_tokens = [
        'ogger created',
        'setting reward',
        'received HELLO',
        'children have connected',
        'child to connect',
        'trigger bcm',
        ' correct poke',
        ' incorrect poke',
        'unknown subject',
        'No trigger found for',
        ]
    
    # Iterate over session lines
    error_encountered = False
    pokes_l = []
    starts_l = []
    stops_l = []

    # This is for keeping track of opto trials on this session
    opto_by_trial_l = []

    for nline, line in enumerate(session_lines):
        # Skip some useless lines
        skip_line = False
        for ignore_token in ignore_tokens:
            if ignore_token in line:
                skip_line = True
        
        if skip_line:
            continue

        # Break on Exceptions
        if "ERROR" in line:
            print("ERROR on line {} of this session, returning None".format(nline))
            error_encountered = True
            break
        
        # Parse useful lines
        elif 'Chosen stim params:' in line or 'Chosen target params:' in line:
            # Get stim params (coded as target params in PokeTrain)
            if 'Chosen stim params:' in line:
                parsed = line.split('Chosen stim params:')[1].split()
            else:
                parsed = line.split('Chosen target params:')[1].split()
            
            # Split out the trial params
            rpi = parsed[1].replace(';', '')
            side = parsed[3].replace(';', '')
            sound = parsed[5].replace(';', '')
            light = parsed[7].replace(';', '')
            
            # Optionally extract these if available
            if 'mean_interval' in line:
                mean_interval = float(parsed[9].replace(';', ''))
            else:
                mean_interval = np.nan
            
            if 'var_interval' in line:
                var_interval = float(parsed[11].replace(';', ''))
            else:
                var_interval = np.nan
            
            # Get the timestamp
            parsed = line.split()
            timestamp_date = line.split()[0]
            timestamp_time = line.split()[1]
            timestamp = timestamp_date + ' ' + timestamp_time
            
            # Store
            starts_l.append((timestamp, rpi, side, sound, light, mean_interval, var_interval))

        elif 'rewarded poke' in line or 'blocked poke' in line:
            # PokeTrain only
            # The port is printed as a dict here, so make sure the keys are ordered
            assert 'from' in line.split()[9]
            rpi_poked = line.split()[10].replace('"', '').replace("'", '').replace(',', '')
            assert 'poke' in line.split()[11]
            side_poked = line.split()[11].replace('"', '').replace("'", '').replace(':', '')
            
            # poke_poked is the combination of rpi and side
            poke_poked = rpi_poked + '_' + side_poked
            
            # timestamp is the first two
            # this is not the "same" timestamp that we're grabbing in the PAFT
            # task, which is problematic 
            # also, needs to be formatted with a T and , instead of space and .
            timestamp = line.split()[0] + 'T' + line.split()[1]
            timestamp = timestamp.replace(',', '.')
            
            # Store
            pokes_l.append((timestamp, poke_poked))
            
        elif ' poke' in line:
            # Poke
            timestamp = line.split()[-3]
            poke_poked = line.split()[-2]
            pokes_l.append((timestamp, poke_poked))
            
            if timestamp == 'target' or poke_poked == 'was':
                1/0
        
        elif 'opto is' in line:
            if 'opto is false' in line:
                opto_by_trial_l.append(False)
            elif 'opto is true' in line:
                opto_by_trial_l.append(True)
            else:
                raise ValueError("unexpected opto line: {}".format(line))
        
        else:
            raise ValueError("unexpected line encountered: {}".format(line))

    # Continue if error session
    if error_encountered:
        pokes_df, starts_df = None, None

    else:
        # Concat
        # These can be empty (length 0)
        pokes_df = pandas.DataFrame.from_records(
            pokes_l, columns=['timestamp', 'port'])
        starts_df = pandas.DataFrame.from_records(
            starts_l, columns=['timestamp', 'rpi', 'side', 'sound', 'light', 'mean_interval', 'var_interval'])
        stops_df = pandas.DataFrame.from_records(
            stops_l, columns=['timestamp', 'port'])  

        # Coerce timestamps
        pokes_df['timestamp'] = pokes_df['timestamp'].apply(
            lambda s: datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f"))

        starts_df['timestamp'] = starts_df['timestamp'].apply(
            lambda s: datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f"))

        # Coerce bool cols
        for bool_col in ['light', 'sound']:
            starts_df[bool_col] = starts_df[bool_col].replace(
                {'True': True, 'False': False}).astype(bool)
        
        # Add opto col onto starts_df
        if len(opto_by_trial_l) > 0:
            assert len(starts_df) == len(opto_by_trial_l)
            starts_df['opto'] = pandas.Series(opto_by_trial_l)
        else:
            # This ensures dtype is bool even if empty
            starts_df['opto'] = pandas.Series(
                [False] * len(starts_df), dtype=bool)
    
    return pokes_df, starts_df

def parse_logfile_lines(all_logfile_lines):
    """Parse the logfile lines to extract poke times.
    
    Returns: session_starts, big_trial_starts_df, big_pokes_df
        session_starts : DataFrame, directly from identify_session_starts
            index : arbitrary integers called 'n_session'
                # These can contain gaps now because poketrain sessions are
                # excluded.
            columns :
                'start_line' : the line number on which that session starts
                'stop_line' : the line number on which that session stops
                'dt_start' : datetime, the time of the session start        
        big_trial_starts_df : DataFrame
            index : MultiIndex
                'n_session' : arbitrary integer
                'trial' : numbered from 0 within each session
                rpi, side : string, the target
                sound, light : bool
        big_pokes_df : DataFrame
            index : MultiIndex
                'n_session' : arbitrary integer
                'poke' : numbered from 0 within each session
            columns :
                'timestamp' : datetime, time of poke
                'port' : string, raw port name from logfile
    """
    ## Split up lines into multiple sessions
    session_starts = identify_session_starts(all_logfile_lines)
    
    # Include only those from paft for now
    #~ session_starts = session_starts[session_starts['task'] == 'paft'].copy()
    
    
    ## Parse each
    pokes_df_keys_l = []
    pokes_df_l = []
    starts_df_l = []
    for n_session in session_starts.index:
        # Get chunk of lines for this session
        session_lines = all_logfile_lines[
            session_starts.loc[n_session, 'start_line']:
            session_starts.loc[n_session, 'stop_line']]

        # Parse that session
        pokes_df, starts_df = parse_lines_from_single_session(
            session_lines)
        
        if pokes_df is None and starts_df is None:
            print("ERROR encountered in session {}, continuing".format(n_session))
            continue
        else:
            assert pokes_df is not None and starts_df is not None
        
        # Store
        pokes_df_l.append(pokes_df)
        starts_df_l.append(starts_df)
        pokes_df_keys_l.append(n_session)

    # Concat results over sessions and pokes/trials
    big_pokes_df = pandas.concat(
        pokes_df_l, keys=pokes_df_keys_l, names=['n_session', 'poke'])
    big_trial_starts_df = pandas.concat(
        starts_df_l, keys=pokes_df_keys_l, names=['n_session', 'trial'])

    return session_starts, big_trial_starts_df, big_pokes_df

def load_data_from_all_mouse_hdf5(mouse_names, munged_sessions,
    path_to_terminal_data='/home/chris/autopilot/data'):
    """Load trial data and weights from HDF5 files for all mice
    
    See load_data_from_single_hdf5 for how the data is loaded from each mouse.
    
    This function then concatenates the results over mice, drops the 
    sessions in `munged_sessions`, nullifies weights where they are saved 
    as 0, and error checks no more than one session per day.
    
    Some redundant and/or useless columns are dropped.
    
    Arguments:
        mouse_names : list
            A list of mouse names. Each should be an HDF5 file in 
            /home/chris/autopilot/data
        munged_sessions : list
            A list of munged session names to drop.
    
    Returns: session_df, trial_data
        session_df : DataFrame
            index : MultiIndex with levels 'mouse' and 'session_name'
                'mouse' : the values in `mouse_names`
                'session_name' : a string like '20210720133116-Male1_0720-Box1'
            columns:
                box : string, the box name
                orig_session_num : int, the original Autopilot session number
                first_trial : datetime, the timestamp of the first trial
                last_trial : datetime, the timestamp of the last trial
                n_trials : number of trials
                approx_duration : last_trial - first_trial
                date : datetime.date, the date of the session
                weight : float, the weight
    
        big_trial_data : DataFrame
            index : MultiIndex with levels mouse, session_name, and trial
                'mouse' : the values in `mouse_names`
                'session_name' : a string like '20210720133116-Male1_0720-Box1'
                'trial' : always starting with zero

            columns :
                'light' : True or False, whether a light was displayed
                'rpi' : One of ['rpi01' ... 'rpi08']
                'side' : One of ['L', 'R']
                'sound' : True or False, whether a sound was played
                'timestamp' : time of trial as a datetime
                    This is directly taken from the 'timestamp' columns in the HDF5
                    file, just decoded from bytes to datetime
                    So I think it is the "start" of the trial
    """
    # Iterate over provided mouse names
    msd_l = []
    mtd_l = []
    keys_l = []
    for mouse_name in mouse_names:
        # Form the hdf5 filename
        #~ h5_filename = '/home/chris/autopilot/data/{}.h5'.format(mouse_name)
        h5_filename = os.path.join(
            path_to_terminal_data, '{}.h5'.format(mouse_name))
        
        # Load data
        mouse_session_df, mouse_trial_data = load_data_from_single_hdf5(
            mouse_name, h5_filename)
        
        # Skip if None
        if mouse_session_df is None and mouse_trial_data is None:
            continue
        else:
            assert mouse_session_df is not None
            assert mouse_trial_data is not None
        
        # Store
        msd_l.append(mouse_session_df)
        mtd_l.append(mouse_trial_data)
        keys_l.append(mouse_name)
    
    # Concatenate
    session_df = pandas.concat(msd_l, keys=keys_l, names=['mouse'])
    trial_data = pandas.concat(mtd_l, keys=keys_l, names=['mouse'])

    # Drop munged sessions
    droppable_sessions = []
    for munged_session in munged_sessions:
        if munged_session in session_df.index.levels[1]:
            droppable_sessions.append(munged_session)
        else:
            print("warning: cannot find {} to drop it".format(munged_session))
    session_df = session_df.drop(droppable_sessions, level='session_name')
    trial_data = trial_data.drop(droppable_sessions, level='session_name')

    
    ## Rename sessions that were saved by the wrong mouse name
    # These tuples are (
    #   the name it was saved as, 
    #   the name it should have been saved as,
    #   the name of the actual mouse that was used)
    rename_sessions_l = [
        ('20210928150737-Female3_0903-Box1', '20210928150737-Female1_0903-Box1', 'Female1_0903'),
        ('20211004125106-Male5_0720-Box2', '20211004125106-Male3_0720-Box2', 'Male3_0720'),
        ('20211005135505-Male4_0720-Box2', '20211005135505-Female4_0903-Box2', 'Female4_0903'),
        ('20211007133256-Male3_0720-Box2', '20211007133256-Male4_0720-Box2', 'Male4_0720'),
        ('20211018165733-Male3_0720-Box2', '20211018165733-Male4_0720-Box2', 'Male4_0720'),
        ('20211106165204-3279-5-Box1', '20211106165204-3279-7-Box1', '3279-7'),
        ('20220309115351-M2_PAFT-Box2', '20220309115351-F2_PAFT-Box2', 'F2_PAFT',),
        ]
    
    # reset index
    trial_data = trial_data.reset_index()
    session_df = session_df.reset_index()
    
    # fix
    for wrong_name, right_name, right_mouse in rename_sessions_l:
        # Fix trial_data
        bad_mask = trial_data['session_name'] == wrong_name
        trial_data.loc[bad_mask, 'session_name'] = right_name
        trial_data.loc[bad_mask, 'mouse'] = right_mouse

        # Fix session_df
        bad_mask = session_df['session_name'] == wrong_name
        session_df.loc[bad_mask, 'session_name'] = right_name
        session_df.loc[bad_mask, 'mouse'] = right_mouse

    # reset index back again
    trial_data = trial_data.set_index(
        ['mouse', 'session_name', 'trial']).sort_index()
    session_df = session_df.set_index(
        ['mouse', 'session_name']).sort_index()


    ## Error check only one session per mouse per day
    n_sessions_per_day = session_df.groupby(['mouse', 'date']).size()
    if not (n_sessions_per_day == 1).all():
        bad_ones = n_sessions_per_day.loc[n_sessions_per_day != 1].reset_index()
        print("warning: multiple sessions in {} case(s):\n{}".format(
            len(bad_ones), bad_ones))
        
        # First example
        bad_mouse = bad_ones['mouse'].iloc[0]
        bad_date = bad_ones['date'].iloc[0]
        bad_sessions = session_df.loc[
            session_df['date'] == bad_date].loc[bad_mouse].T
        print("remove one of these sessions:\n{}".format(bad_sessions))

    # Nullify zero weights
    session_df.loc[session_df['weight'] < 1, 'weight'] = np.nan
    
    # Drop useless columns from session_df
    session_df = session_df.drop(
        ['weights_date_string', 'weights_dt_start'], axis=1)
    
    # Drop columns from trial_data that are redundant with session_df
    # (because this is how they were aligned)
    trial_data = trial_data.drop(
        ['orig_session_num', 'box', 'date'], axis=1)

    # Return
    return session_df, trial_data

def load_data_from_single_hdf5(mouse_name, h5_filename):
    """Load session and trial data from a single mouse's HDF5 file
    
    The trial data and the weights are loaded from the HDF5 file. The
    columns are decoded and coerced into more meaningful dtypes. Some
    useless columns are dropped. The box is inferred from the rpi names.
    
    The trial data and weights are aligned based on the original session
    number and the inferred box. This seems to work but is not guaranteed to 
    do so. A unique, sortable session_name is generated.
    
    The same session_names are in both returned DataFrames. 
    No sessions are dropped.
    
    Arguments:
        mouse_name : string
            Used to name the sessions
        h5_filename : string
            The filename to a single mouse's HDF5 file
    
    Returns: mouse_session_df, mouse_trial_data
        None, None if the hdf5 file can't be loaded
    
        Both of the following are sorted by their index.
        mouse_session_df : DataFrame
            index : string, the session name
                This is like 20210914135527-Male4_0720-Box2, based on the
                time of the first trial.
            columns: 
                box : string, the box name
                orig_session_num : int, the original Autopilot session number
                first_trial : datetime, the timestamp of the first trial
                last_trial : datetime, the timestamp of the last trial
                n_trials : number of trials
                approx_duration : last_trial - first_trial
                date : datetime.date, the date of the session
                weights_date_string : string, the date as a string use in the
                    weights field of the HDF5 file
                weight : float, the weight
                weights_dt_start : datetime, the datetime stored in the
                    weights field of the HDF5 file
        
        mouse_trial_data : DataFrame
            index : MultiIndex
                session_name : string, the unique session name
                trial : int, the trial number starting from 0
            columns :
                light, sound : bool, whether a  light or sound was played
                rpi : string, the name of the target rpi on that trial
                orig_session_num : int, the original Autopilot session number
                side : 'L' or 'R'
                timestamp : the trial time, directly from the HDF5 file
                box : 'Box1' or 'Box2', inferred from rpi name
                date : datetime.date, the date of the session
    """
    ## Load trial data and weights
    cannot_load = False
    try:
        with tables.open_file(h5_filename) as fi:
            mouse_trial_data = pandas.DataFrame(
                fi.root['data']['PAFT_protocol']['S00_PAFT']['trial_data'][:])
            mouse_weights = pandas.DataFrame(
                fi.root['history']['weights'][:])
    except tables.HDF5ExtError:
        cannot_load = True
    
    if cannot_load:
        print("cannot load {}".format(h5_filename))
        return None, None
    

    ## Coerce dtypes for mouse_trial_data
    # Columns
    # light, rpi, side, sound : bytes
    # timestamp: time of trial start (?) as bytes
    # session: session number, starting with 1
    # trial, trial_num, trials_total : none are what we want

    # Decode columns that are bytes
    for decode_col in ['light', 'rpi', 'side', 'sound', 'timestamp']:
        mouse_trial_data[decode_col] = mouse_trial_data[decode_col].str.decode('utf-8')

    # Coerce timestamp to datetime
    mouse_trial_data['timestamp'] = mouse_trial_data['timestamp'].apply(
        lambda s: datetime.datetime.fromisoformat(s))

    # Hack
    assert 'opto' not in mouse_trial_data.columns
    mouse_trial_data['opto'] = mouse_trial_data['light'].copy()

    # Coerce the columns that are boolean
    bool_cols = ['sound']
    for bool_col in bool_cols:
        mouse_trial_data[bool_col] = mouse_trial_data[bool_col].replace(
            {'True': True, 'False': False}).astype(bool)
    
    # Hack
    mouse_trial_data['light'] = mouse_trial_data['light'].replace(
        {'True': True, 'False': False, 'opto_on': False, 'opto_off': False}
        )
    assert mouse_trial_data['light'].isin([True, False]).all()
    mouse_trial_data['light'] = mouse_trial_data['light'].astype(bool)
    
    # Hack
    mouse_trial_data['opto'] = mouse_trial_data['opto'].replace(
        {'True': False, 'False': False, 'opto_on': True, 'opto_off': False}
        ).astype(bool)  
    assert mouse_trial_data['opto'].isin([True, False]).all()
    mouse_trial_data['opto'] = mouse_trial_data['opto'].astype(bool)
    
    
    ## Coerce dtypes for mouse_weights
    # Rename more meaningfully
    mouse_weights = mouse_weights.rename(columns={'date': 'date_string'})
    
    # Decode
    mouse_weights['date_string'] = mouse_weights['date_string'].str.decode('utf-8')

    # Convert 'date_string' to Timestamp
    mouse_weights['dt_start'] = mouse_weights['date_string'].apply(
        lambda s: datetime.datetime.strptime(s, '%y%m%d-%H%M%S'))
    
    # Calculate raw date (dropping time)
    mouse_weights['date'] = mouse_weights['dt_start'].apply(
        lambda dt: dt.date())
    
    
    ## Drop useless columns
    # Drop columns for mouse_trial_data that I don't understand anyway
    mouse_trial_data = mouse_trial_data.drop(
        ['trial', 'trial_num', 'trials_total'], axis=1)
    
    # Drop unused columns
    mouse_weights = mouse_weights.drop('stop', axis=1)

    # Rename meaningfully
    mouse_weights = mouse_weights.rename(columns={
        'start': 'weight',
        'session': 'orig_session_num',
        })

    
    ## Asign 'box' based on 'rpi'
    mouse_trial_data['box'] = '???'
    box1_mask = mouse_trial_data['rpi'].isin(
        ['rpi01', 'rpi02', 'rpi03', 'rpi04'])
    box2_mask = mouse_trial_data['rpi'].isin(
        ['rpi05', 'rpi06', 'rpi07', 'rpi08'])
    mouse_trial_data.loc[box1_mask, 'box'] = 'Box1'
    mouse_trial_data.loc[box2_mask, 'box'] = 'Box2'
    assert mouse_trial_data['box'].isin(['Box1', 'Box2']).all()

    
    ## Identify sessions
    # Sometimes the session number restarts numbering and I don't know why
    # Probably after every box change? Unclear
    # Let's group by ['box', 'date', 'session'] and assume that each
    # is unique. This could still fail if the renumbering happens without
    # a box change, somehow.
    # This will also fail if a session spans midnight
    
    # First add a date string
    mouse_trial_data['date'] = mouse_trial_data['timestamp'].apply(
        lambda dt: dt.date())
    
    # Group by ['box', 'date', 'session']
    gobj = mouse_trial_data.groupby(['box', 'date', 'session'])
    
    # Extract times of first and last trials
    session_df = pandas.DataFrame.from_dict({
        'first_trial': gobj['timestamp'].min(),
        'last_trial': gobj['timestamp'].max(),
        'n_trials': gobj['timestamp'].size(),
        })
    
    # Calculate approximate duration and do some basic sanity checks
    session_df['approx_duration'] = (
        session_df['last_trial'] - session_df['first_trial'])
    assert (
        session_df['approx_duration'] < datetime.timedelta(hours=2)).all()
    assert (
        session_df['approx_duration'] >= datetime.timedelta(seconds=0)).all()
    
    # Reset index, and preserve the original session number
    session_df = session_df.reset_index()
    session_df = session_df.rename(columns={'session': 'orig_session_num'})

    # Extract the date of the session
    session_df['date'] = session_df['first_trial'].apply(
        lambda dt: dt.date())

    # Create a unique, sortable session name
    # This will generate a name like 20210907145607-Female4_0903-Box1
    # based on the time of the first trial
    session_df['session_name'] = session_df.apply(
        lambda row: '{}-{}-{}'.format(
        row['first_trial'].strftime('%Y%m%d%H%M%S'), mouse_name, row['box']),
        axis=1)
    
    
    ## Align session_df with mouse_weights
    # The assumption here is that each is unique based on 
    # ['date', 'orig_session_num']. Otherwise this won't work.
    assert not session_df[['date', 'orig_session_num']].duplicated().any()
    assert not mouse_weights[['date', 'orig_session_num']].duplicated().any()

    # Rename the weights columns to be more meaningful after merge
    mouse_weights = mouse_weights.rename(columns={
        'date_string': 'weights_date_string',
        'dt_start': 'weights_dt_start',
        })
    
    # Left merge, so we always have the same length as session_df
    # This drops extra rows in `mouse_weights`, corresponding to sessions
    # with no trials, typically after a munging event, which is fine
    session_df = pandas.merge(
        session_df, mouse_weights, how='left', 
        on=['date', 'orig_session_num'])
    
    # Make sure there was an entry in weights for every session
    assert not session_df.isnull().any().any()


    ## Add the unique session_name to mouse_trial_data
    # Get 'date' to align with session_df
    # This will fail if the session crossed over midnight
    mouse_trial_data['date'] = mouse_trial_data['timestamp'].apply(
        lambda dt: dt.date())
    mouse_trial_data = mouse_trial_data.rename(
        columns={'session': 'orig_session_num'})
    
    # Align on these columns which should uniquely defined
    join_on = ['box', 'date', 'orig_session_num']
    mouse_trial_data = mouse_trial_data.join(
        session_df.set_index(join_on)['session_name'],
        on=join_on)
    assert not mouse_trial_data['session_name'].isnull().any()
    
    # Create a 'trial' column, numbering within each session
    res_l = []
    res_keys_l = []
    for session_name, sub_mtd in mouse_trial_data.groupby('session_name'):
        # Check that it's sorted
        diffs = mouse_trial_data['timestamp'].diff().dropna()
        assert (diffs > datetime.timedelta(0)).all()
        
        # Drop column because it's a key now
        res = sub_mtd.drop('session_name', axis=1)
        
        # Add trial column
        res['trial'] = range(len(res))
        res = res.set_index('trial')
        
        # Store
        res_l.append(res)
        res_keys_l.append(session_name)
    
    # Concat the renumbered results
    renumbered_mouse_trial_data = pandas.concat(
        res_l, keys=res_keys_l, names=['session_name'])

    # Index session_df by session_name
    session_df = session_df.set_index('session_name')
    
    # Sort both
    renumbered_mouse_trial_data = renumbered_mouse_trial_data.sort_index()
    session_df = session_df.sort_index()
    
    # Error check
    assert not session_df.index.duplicated().any()
    assert not renumbered_mouse_trial_data.index.duplicated().any()
    assert renumbered_mouse_trial_data.index.levels[0].equals(
        session_df.index)

    return session_df, renumbered_mouse_trial_data

