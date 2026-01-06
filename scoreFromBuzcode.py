import os, h5py
import pandas as pd, numpy as np
from lfp.readLFP import *
from utils.getDataFiles import *

def get_state_intervals_df(basepath, lfp_filepath=None, 
                            sleepscore_mat=None, num_channels=1): 
    if sleepscore_mat is None:
        sleepscore_mat = os.path.join(basepath, f'{os.path.basename(basepath)}.SleepState.states.mat')
        print('Found sleepscore file')
    if not os.path.exists(sleepscore_mat):
        raise ValueError(f'Provided sleepscore states file not provided or invalid! Run sleep_score_allrecs function on list of \
                         recording paths to get output.')
    
    if lfp_filepath is None:
        lfp_filepath = os.path.join(basepath, f'{os.path.basename(basepath)}.lfp')        
    state_intervals_df = pd.DataFrame({'start': [], 'end': [], 'state': []})

    with h5py.File(sleepscore_mat, "r") as f:
        sleep = f["SleepState"]
        # --- per-bin info ---
        idx = sleep["idx"]
        timestamps = idx["timestamps"][0]   # shape (T,)
        states = idx["states"][0]          # shape (T,)
        print(np.array(timestamps).shape, np.array(states).shape)
        # map numeric codes -> names (for your file: 1=WAKE, 3=NREM)
        #code_to_name = {1.0: "WAKE", 3.0: "NREM", 5.0:"REM"}
        #state_name = np.array([code_to_name.get(float(s), "UNKNOWN") for s in states])
        
        # --- interval info helper ---
        ints = sleep["ints"]
        wake_df = load_interval_arrays(ints['WAKEstate'][()], 'WAKE')
        nrem_df = load_interval_arrays(ints['NREMstate'][()], 'NREM')
        rem_df  = load_interval_arrays(ints['REMstate'][()], 'REM')

        state_intervals_df = pd.DataFrame(
            np.vstack([wake_df, nrem_df, rem_df]),
            columns=['start', 'end', 'state'])
        state_intervals_df['start'] = state_intervals_df['start'].astype(float)
        state_intervals_df['end'] = state_intervals_df['end'].astype(float)
        # adjust epoch lengths
        if not os.path.exists(lfp_filepath):
            raise ValueError(f'{lfp_filepath} is not a valid LFP filepath!!')
        else:
            ### GET THE EPOCH LENGTH FROM SLEEPSCORE OUTPUT
            epoch_len = sleepscore_epoch_from_lfp(lfp_filepath, timestamps, num_channels=num_channels)
        state_intervals_df['start'] = state_intervals_df['start'].apply(lambda time: time * epoch_len)
        state_intervals_df['end'] = state_intervals_df['end'].apply(lambda time: time * epoch_len)
        state_intervals_df['duration'] = state_intervals_df['end'] - state_intervals_df['start'] 
        state_intervals_df.sort_values(by='start', inplace=True)
        return state_intervals_df, timestamps        

def get_state_at_time(time, state_intervals_df, basepath = None):
    if state_intervals_df is None and basepath is not None:
        state_intervals_df = get_state_intervals_df(basepath)
    return state_intervals_df[(state_intervals_df['start'] < time
                                ) & (state_intervals_df['end'] > time)]['state'].iloc[0]
    
def flatten_state_ints_df(state_ints_df):
    row_idx = 0
    flattened_df = pd.DataFrame({'time':[], 'state': [], 'duration': []})
    for row_idx in range(len(state_ints_df)):
        row = state_ints_df.iloc[row_idx]
        start = row['start']
        end = row['end']
        state = row['state']
        duration = row['duration']
        new_range = pd.Series(np.arange(start, end))
        len_range = len(new_range)
        new_rows = pd.DataFrame({'time': new_range, 'state': [state] * len_range, 'duration': [duration] * len_range})
        flattened_df = pd.concat([flattened_df, new_rows])
    return flattened_df

def load_sleep_state_df(path):
    basename = os.path.basename(path)
    df_path = os.path.join(path, f'{basename}_sleep_state_df.parquet')
    if not os.path.exists(df_path):
        print('No sleep state dataframe found! run get_state_intervals_df() and flatten_state_ints_df')
        return None
    else:
        sleep_state_df = pd.read_parquet(df_path)
        return sleep_state_df

def load_interval_arrays(sleep_intervals_arr, label):
    arr = np.array(sleep_intervals_arr)

    # Empty
    if arr.size == 0:
        arr2 = arr.reshape(0, 2)

    # Single interval stored as shape (2,)
    elif arr.ndim == 1:
        if arr.size != 2:
            raise ValueError(f"Unexpected interval shape {arr.shape}")
        arr2 = arr.reshape(1, 2)

    # Standard shape (2, N)
    elif arr.ndim == 2 and arr.shape[0] == 2:
        arr2 = arr.T

    else:
        raise ValueError(f"Unexpected state array shape {arr.shape}")

    labels = np.full((arr2.shape[0], 1), label)
    return np.concatenate((arr2, labels), axis=1)

def sleepscore_epoch_from_lfp(lfp_path, timestamps, num_channels=1):
    lfp = load_lfp(lfp_path, num_channels=num_channels)          # your loader
    n_channels = num_channels if num_channels == 1 else lfp.shape[1]

    fs_lfp = 1250                     # Hz
    n_epochs = len(timestamps)        # 610 from SleepState.idx.timestamps

    # Correct: use bytes and divide by channels
    n_bytes = os.path.getsize(lfp_path)
    n_total_int16 = n_bytes // 2              # 2 bytes per int16
    n_samples_per_channel = n_total_int16 // n_channels

    duration_sec = n_samples_per_channel / fs_lfp
    epoch_len = duration_sec / n_epochs
    return epoch_len
