import traceback
from utils.getDataFiles import *
from utils.alignmentFunctions import get_analog_times, align_scope_triggers_to_frames, motion_to_2p_bins, categorical_to_2p
from intan.readIntan import *
from twop.getSuite2POutput import *
from state.getFacemapData import *

class TwoPData:
    # PROVIDE ANALOG CHANNELS AS NUM ADC CHANNEL (0-8), NOT ANALOG/AUX TOTAL
    '''
    Class to encapsulate ALL recording data 

    Parameters:
        suite2p_basepath (str or Path-like): filepath where suite2p data is located 
        
        intan_basepath (str or Path-like): filepath where intan data is located (*required: info.rhd and amp_analog_aux_int.dat.. files)

        scope_fps (float; default=None): sampling rate of 2P scope

        twop_channel (int; default=2): recording channel of scope

        pd_channel (int; default=None): channel of photodiode

        camera_channel (int; default=None): channel of camera

        treadmill_channel (int; default=None): channel of treadmill

        experiment_XML (str or Path-like): file containing imaging rate and experiment metadata
    '''
    def __init__(self, suite2p_basepath, intan_basepath, facemap_path=None, scope_fps=None,
                  twop_channel=2, pd_channel=None, camera_channel=None, treadmill_channel=None, 
                  plot_intan=False, intan_plot_start_s=None, intan_plot_end_s=None):
        # datapath
        self.suite2p_basepath = suite2p_basepath
        self.intan_basepath = intan_basepath
        self.facemap_path = facemap_path

        # recording channel info
        self.twop_chan = twop_channel
        self.pd_chan = pd_channel
        self.camera_chan = camera_channel
        self.treadmill_chan = treadmill_channel
        if not os.path.exists(self.suite2p_basepath):
            raise ValueError(f'Suite2p path {self.suite2p_basepath} does not exist.')
        
        self.s2p_out = Suite2POutput(self.suite2p_basepath, scope_fs=scope_fps)
        self.scope_fps = self.s2p_out.scope_fs

        (self.fs_intan, self.photodiode_raw, self.twop_raw, self.camera_raw, self.treadmill_raw) = get_all_intan_data(self.intan_basepath,
                                                                                        twop_chan=twop_channel, 
                                                                                        pd_chan=pd_channel,
                                                                                        camera_chan=camera_channel, 
                                                                                        treadmill_chan=treadmill_channel,
                                                                                        plot_intan=plot_intan,
                                                                                          intan_plot_start_s= intan_plot_start_s,
                                                                                            intan_plot_end_s= intan_plot_end_s)  
        ## get times in seconds of TTL triggers for scope, photodiode, camera, and treadmill
        self.scope_times, self.scope_times_end = get_analog_times(self.twop_raw, upTransition=True, signal_name='scope')
        try:
            self.scope_times, self.scope_times_end = align_scope_triggers_to_frames(self.s2p_out, self.scope_times)
        except:
            print(f'Did not realign scope times')
        if self.photodiode_raw is not None:
            self.pd_times, self.pd_times_end = get_analog_times(self.photodiode_raw, signal_name='photodiode')
        if self.camera_chan is not None:
            self.camera_times, self.camera_times_end = get_analog_times(self.camera_raw, signal_name='camera')
        if self.treadmill_chan is not None:
            self.treadmill_times, self.treadmill_times_end = get_analog_times(self.treadmill_raw, signal_name='treadmill') 
        # read in facemap data
        if self.facemap_path is not None:
            self.facemap_data = get_facemap_data(self.facemap_path)

    def make_frame_df(self, output_csv=False, output_filepath=None):
        """
        Create a dataframe of relative time estimates from trigger and raw frame times in seconds since recording start

        Args:
            output_csv (bool): whether to output a csv  
            output_filepath (str, Path-like):
        Returns:
            scope_df (pd.DataFrame): dataframe of raw scope times 
        """
        timeEst = np.arange(self.s2p_out.nframes) / self.s2p_out.scope_fs
        scope_df = pd.DataFrame({'timeEst':timeEst, 'frame_time':self.scope_times})
        if output_csv:
            self.df_to_csv(scope_df, output_filepath= output_filepath)
        return scope_df
    
    def make_state_df(self, cam_fps=None, smoothing_kernel=5, movement_percentile=70, min_duration_s=1.0,
                      annotate_state=True,motion_indices={'motion':1}, pupil_data=False, paw_data=False,
                         annotate_state_with_pupil=False, output_csv=False, output_filepath=None, use_motsvd = False, min_max_norm=True):
        """
        Create state dataframe (from getFacemapData.py) using data from base folder

        Args:
            cam_fps (int; default is 30): camera ttl rate

            treadmill_data (bool; default is True): whether treadmill data was recorded

            smoothing_kernel (int): smoothing factor for treadmill signal

            movement_percentile (int; default=70): percentile for movement detection

            output_csv (bool): whether to output a csv file, if so provide a path to the output_filepath argument 

            output_filepath (str, Path-like): path to save CSV file to if output_csv == True

        Returns:
            state_dataframe (pd.DataFrame): dataframe containing camera aligned timestamps, treadmill, motion, and pupil signals
        """
        if cam_fps is None:
            cam_fps = get_camera_fps(self.camera_times)
        state_dataframe = get_state_df(facemap_data= self.facemap_data,
                                camera_times= self.camera_times,
                                 treadmill_signal= self.treadmill_raw,
                                    cam_fps=cam_fps, pupil_data=pupil_data, paw_data=paw_data,
                                      smoothing_kernel=smoothing_kernel, 
                                      min_duration_s=min_duration_s,
                                      movement_percentile= movement_percentile,
                                      annotate_state=annotate_state,
                                      annotate_state_with_pupil=annotate_state_with_pupil,
                                      motion_indices=motion_indices,# what index is motion saved in facemap,
                                      use_motsvd = use_motsvd,
                                       min_max_norm=min_max_norm)
        if output_csv:
            self.df_to_csv(state_dataframe, output_filepath)
        return state_dataframe
    
    def frame_state_df(self, state_df, output_csv=False, output_filepath= None, tolerance=None):
        """
        Create dataframe of camera times, scaled state data, and nearest frames

        Args:  
            state_df (pd.DataFrame): 

            tolerance (float ; default is None): 

        Returns:
            frame_state_df (pd.DataFrame)
        
        """
        # create a copy of state dataframe for modification
        statedf_copy = state_df.copy()

        # min max normalize pupil area and motion
        motion_scaled = MinMaxScaler().fit_transform(statedf_copy['motion'].to_numpy().reshape(-1, 1)).flatten()
        motion_bool= statedf_copy['motion_bool']
        # drop curr cols and replace with minmax normalized 
        #statedf_copy.drop(['motion_raw', 'motion_smooth', 'pupil_area', 'treadmill_raw'], axis=1, inplace=True)

        # create dataframe for used merging to state vals/cam times
        frame_df = pd.DataFrame({'nearest_frame_idx': np.arange(self.s2p_out.nframes),
                                  'frame_start_time': self.scope_times}).reset_index()
        # merge on nearest 
        nearest_frames = pd.merge_asof(statedf_copy[['time']],
                                        frame_df[['nearest_frame_idx', 'frame_start_time']],
                                          left_on='time', right_on='frame_start_time',
                                            direction='forward', tolerance=tolerance)
        frame_state_df = nearest_frames
        
        data_to_add = {'motion_bool': motion_bool,
                       'motion': motion_scaled}
        
        state_cols = list(statedf_copy.columns)
        if 'pupil_area' in state_cols:
            pupil_scaled = MinMaxScaler().fit_transform(statedf_copy['pupil_area'].to_numpy().reshape(-1, 1)).flatten()
            data_to_add['pupil'] = pupil_scaled
        if 'treadmill' in state_cols:
            treadmill_scaled = MinMaxScaler().fit_transform(statedf_copy['treadmill'].to_numpy().reshape(-1, 1)).flatten()
            data_to_add['treadmill'] = treadmill_scaled
            data_to_add['locomotion_bool'] = statedf_copy['locomotion_bool']

        frame_state_df = pd.concat([frame_state_df, pd.DataFrame(data_to_add)], axis=1)

        frame_state_df['nearest_frame_idx'] = frame_state_df['nearest_frame_idx'].fillna(-1).astype(int)

        if output_csv:
            self.df_to_csv(frame_state_df, output_filepath=output_filepath)
        return frame_state_df

    def make_aligned_frame_df(self, state_df):
        motion_2p = motion_to_2p_bins(self.scope_times, state_df['motion'], state_df['time'])
        state_2p = categorical_to_2p(self.scope_times, state_df['state'], state_df['time'])
        aligned_df = pd.DataFrame({
            "frame_time": self.scope_times,
            "motion": motion_2p,
            "state": state_2p
        })
        return aligned_df

    
def df_to_csv(self, dataframe, output_filepath):
    """
    Helper function to save dataframes to a csv file if output_csv = True

    Args:
        dataframe (pd.DataFrame): pandas dataframe to export

        output_filepath (str, Path-like): output path to save dataframe csv. if None, put into data basepath folder

    Returns:
        None, saves dataframe to output_filepath
    """
    if output_filepath is None:
        output_filepath = os.path.join(self.intan_basepath)
    if not output_filepath.endswith('.csv'):
        output_filepath += '.csv'
    try:
        dataframe.to_csv(output_filepath)
    except:
        traceback.print_exc()
