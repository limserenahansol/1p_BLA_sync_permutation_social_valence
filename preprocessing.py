import pandas as pd
import numpy as np
import h5py as h5
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import os
from os import path
import traceback
from copy import deepcopy
import logging.config
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
from datetime import datetime
# from util import rnorm



_folder_sniffer = lambda fdir: [i for i in os.listdir(fdir) if
                                os.path.isdir(f'{fdir}//{i}') and len(i.split('_')) == 2]

def create_logger(logdir):
    global prep_logger
    DATETIME = datetime.now().strftime('%Y%m%d%H%M%S')
    LOG_LVL_LOOKUP_TABLE = {
        "DEBUG": DEBUG,
        "INFO": INFO,
        "WARNING": WARNING,
        "ERROR": ERROR,
        "CRITICAL": CRITICAL,
    }

    DEFAULT_LOGGER_CONFIG = {
        'version': 1,
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO'
            }
        },
        'root': {
            'handlers': ['console'],
            'level': 'DEBUG'
        }
    }

    DEFAULT_LISTENER_CONFIG = {
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': {
            'detailed': {
                'class': 'logging.Formatter',
                'format': '%(asctime)-4s %(levelname)-8s  %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'detailed',
                'level': 'INFO'
            },
            'file': {
                'class': 'logging.FileHandler',
                'filename': logdir+'/'+DATETIME + '.log',
                'mode': 'w',
                'formatter': 'detailed',
                'level': 'DEBUG'
            },
            'errors': {
                'class': 'logging.FileHandler',
                'filename': logdir+'/'+DATETIME + '-errors.log',
                'mode': 'w',
                'formatter': 'detailed',
                'level': 'WARNING'
            }
        },
        'root': {
            'handlers': ['console', 'file', 'errors'],
            'level': 'DEBUG'
        }
    }
    logging.config.dictConfig(DEFAULT_LISTENER_CONFIG)
    prep_logger = logging.getLogger('PrepLogger')
    prep_logger.setLevel(DEBUG)

class ProjectScreener:
    _name = "ProjectScreener"
    _PROJ_INFO_FDIR = 'ProjectInfo'
    _DATETIME_PATTERN = '%Y%m%d%H%M%S'


    def __init__(self, proj_dir, author, proj_info_fn=None):
        self.author = author
        self.project_dir = proj_dir
        self._processed_file = []
        self._failed_file = []
        self._proj_info_db = None
        self._modified_from = 'VOID'
        self._session_id = datetime.now().strftime('%Y%m%d%H%M%S')
        self._proj_info_attrs = {}
        self._check_proj_info_dir(proj_info_fn)
        create_logger(proj_dir+'ProjectInfo/')

    def _check_proj_info_dir(self, proj_info_fn):
        if self._PROJ_INFO_FDIR in os.listdir(self.project_dir):
            proj_info_folder = self.project_dir + self._PROJ_INFO_FDIR
            if not os.path.isdir(proj_info_folder):
               prep_logger.error(
                    f"Please rename the file(s) in the project folder with the name {self._PROJ_INFO_FDIR} as it may disrupt the folder structure recognition process.")
            if proj_info_fn is None:
                proj_info_fid_list = [datetime.strptime(i[:-3], self._DATETIME_PATTERN) for i in proj_info_folder if
                                      i.endswith('.h5')]
                if len(proj_info_fid_list)>0:
                    proj_info_fn = min(proj_info_fid_list).strftime(self._DATETIME_PATTERN) + '.h5'  # Find the latest file
                    self.load_proj_info_file(proj_info_fn)
            else:
                self.load_proj_info_file(proj_info_fn)
        else:
           prep_logger.error(
                f'No project infomation folder ({self._PROJ_INFO_FDIR}) found under the directory {self.project_dir}')

    def load_proj_info_file(self, proj_info_fn, allow_corrupted=False):
        prep_logger.info(f'[{self._name}] Loading project info file "{proj_info_fn}"')

        proj_info_fdir = f'{self.project_dir}/{self._PROJ_INFO_FDIR}/{proj_info_fn}'
        self._proj_info_db = pd.read_hdf(proj_info_fdir)

        self._proj_info_attrs['file name'] = proj_info_fn
        self._proj_info_attrs['fid'] = proj_info_fn[:-3]
        with h5.File(proj_info_fdir, 'r') as hf:
            self._proj_info_attrs['ID'] = hf.attrs['ID']
            self._proj_info_attrs['MODIFIED_BY'] = hf.attrs['MODIFIED_BY']
            self._proj_info_attrs['MODIFIED_FROM'] = hf.attrs['MODIFIED_FROM']
            self._modified_from = hf.attrs['MODIFIED_FROM']
            self._proj_info_attrs['CORRUPTED'] = hf.attrs['CORRUPTED']

        if not allow_corrupted:
            if self._proj_info_attrs['CORRUPTED']:
                # If unexpected corrupted file, reset everything
                self._proj_info_db = None
                self._proj_info_attrs = {}
                self._modified_from = 'VOID'
                prep_logger.info(f'[{self._name}] No file loaded because the selected project info file is corrupted!')

    def scan_proj_subfolder(self):
        tmp_dir = self.project_dir
        prep_logger.info(f'[{self._name}] Scanning project folder: {tmp_dir}   ')
        animal_folders = _folder_sniffer(tmp_dir)
        db = pd.DataFrame(
            columns=['genotype', 'animal_id', 'exp_type', 'exp_id', 'session_id', 'folder_dir',
                     'ca_fn', 'gpio_fn', 'behavior_fn', 'preprocessed'])
        for ia in animal_folders:
            i_animal_dir = self.project_dir + '/' + ia + '/'
            genotype, animal_id = ia.split('_')
            exp_folders = _folder_sniffer(i_animal_dir)
            for ie in exp_folders:
                i_exp_dir = self.project_dir + '/' + ia + '/' + ie + '/'
                exp_type, exp_id = ie.split('_')
                session_folder = _folder_sniffer(i_exp_dir)
                for i_session in session_folder:
                    i_session_dir = self.project_dir + '/' + ia + '/' + ie + '/' + i_session + '/'
                    if i_session.startswith('session'):
                        session_id = i_session.split('_')[1]
                        file_list = [i for i in os.listdir(i_session_dir) if os.path.isfile(i_session_dir + i)]
                        file_dict = {'genotype': genotype, 'animal_id': int(animal_id), 'exp_type': exp_type,
                                     'exp_id': str(exp_id), 'session_id': int(session_id),'folder_dir': i_session_dir,
                                     'ca_fn': 'not found', 'gpio_fn': 'not found', 'behavior_fn': 'not found',
                                     'preprocessed': -2}  # preprocessed = -2 if not scanned, just for debug
                        for i in file_list:
                            i_ftype, i_ext = i.lower().split('_')[-1].split('.')
                            if i_ftype == 'ca' and i_ext in ['csv', 'xls', 'xlsx']:
                                file_dict['ca_fn'] = i
                            elif 'gpio' in i_ftype  and i_ext in ['csv', 'xls', 'xlsx']:
                                file_dict['gpio_fn'] = i
                            elif i_ftype in ['behavior', 'fc'] and i_ext in ['csv', 'xls', 'xlsx']:
                                file_dict['behavior_fn'] = i
                        if 'preprocessed_data.h5' in file_list:
                            file_dict['preprocessed'] = 1  # 1 if preprocessed
                        else:
                            file_dict['preprocessed'] = 0  # 0 if not processed
                        if len([i for i in ['ca_fn', 'gpio_fn', 'behavior_fn'] if file_dict[i] is None]) > 0:
                            file_dict['preprocessed'] = -1  # -1 if cannot be processed because of missing files
                        db.loc[db.shape[0]] = file_dict

        self._proj_info_db = db
        prep_logger.info('Finished folder scanning\n')

    def batch_processing(self, reprocess_all=False):
        if self._proj_info_db is None:
            self.scan_proj_subfolder()
        fdir_to_be_processed = self._proj_info_db.loc[self._proj_info_db['preprocessed'] == 0, 'folder_dir'].to_list()
        if reprocess_all:
            fdir_to_be_processed.extend(self._proj_info_db.loc[self._proj_info_db['preprocessed'] == 1, 'folder_dir'].to_list())
        n = 0
        for fdir in fdir_to_be_processed:
            n += 1
            prep_logger.info(f'[{self._name}] Processing  {n}/{len(fdir_to_be_processed)}...')
            try:
                align_ca_behav_data(fdir, self._session_id)
                self._processed_file.append(fdir)
            except:
                self._failed_file.append(fdir)
                prep_logger.info(f'[{self._name}] Error when processing: {fdir}\n{traceback.format_exc()}')

        self.write_proj_info_file(self.author)

    def write_proj_info_file(self, author):
        new_file_dir = f'{self.project_dir}/{self._PROJ_INFO_FDIR}/{self._session_id}.h5'

        if self._proj_info_db is None:
            self.scan_proj_subfolder()

        with h5.File(new_file_dir, 'w') as hf:
            hf.attrs['ID'] = self._session_id
            hf.attrs['MODIFIED_BY'] = author
            hf.attrs['CORRUPTED'] = False
            hf.attrs['MODIFIED_FROM'] = self._modified_from
            hf.attrs['PROCESSED_FILE'] = str(self._processed_file)
            hf.attrs['FAILED_FILE'] = str(self._failed_file)

        self._proj_info_db.to_hdf(new_file_dir, 'a')

        prep_logger.info(f'[{self._name}] New project info file has been created ({new_file_dir})')


def csv_loader(fn, **kwargs):
    _name = 'Excel file loader'
    prep_logger.info(f'[{_name}] Checking file validity...')
    if path.exists(fn):
        extension_name = fn.split('.')[-1]
        if extension_name == 'xlsx' or extension_name == 'xls':
            ftype = 'excel'
        elif extension_name == 'csv':
            ftype = 'csv'
        else:
            prep_logger.error(f"Unknown file type {extension_name}")

        # Load excel/csv and standardize them into the class attribute "ca_timeline"
        prep_logger.info(f'[{_name}] Loading file...')
        if ftype == 'excel':
            ftable = pd.read_excel(fn, engine='openpyxl', **kwargs)
        elif ftype == 'csv':
            ftable = pd.read_csv(fn, **kwargs)
        return ftable
    else:
        prep_logger.error(f'\n[{_name}] File not found: {fn}')
        return None


def load_ca_data(ca_data_fn):
    _name = 'Load Ca Data'
    prep_logger.info(f'[{_name}] Importing Ca data file...')
    extension_name = ca_data_fn.split('.')[-1]
    file_postfix = ca_data_fn.strip(' ').split('_')[-1][:-(len(extension_name) + 1)].lower()
    expected_postfix = 'ca'
    if file_postfix == expected_postfix:
        ca_data = csv_loader(ca_data_fn)
        cell_accepted = ca_data.iloc[:, 1:].columns[['accepted' in j for j in ca_data.iloc[0, 1:]]].to_list()
        ca_data = ca_data.iloc[1:, :].astype(np.float32)
        ca_data = ca_data.set_index(ca_data.columns[0])
        ca_data = ca_data.sort_index()  # sort rows by time
    else:
        prep_logger.error(f'\n[{_name}] Incorrect file postfix: the file name should be end with "{expected_postfix}", '
                        f'instead it is {file_postfix}')
    prep_logger.info(f'[{_name}] Ca data file imported.')
    return ca_data, cell_accepted


def exp_folder_loader(fdir):
    flist = os.listdir(fdir)
    behav_fn, ca_tl_fn, ca_data_fn = [None, None, None]
    for i_fn in flist:
        i_ftype = i_fn.split('_')[-1].split('.')[0].lower()
        if i_ftype in ['behavior', 'fc']:
            behav_fn = f'{fdir}/{i_fn}'
        elif 'gpio' in i_ftype:
            ca_tl_fn = f'{fdir}/{i_fn}'
        elif i_ftype == 'ca':
            ca_data_fn = f'{fdir}/{i_fn}'
    return behav_fn, ca_tl_fn, ca_data_fn


class TimeInterpolator:
    _name = 'TimeInterpolator'
    _DEFAULT_INTERP_CONFIG = {
        'kind': 'linear',
        'bounds_error': False,
        'fill_value': np.nan,
    }

    def __init__(self, timeline_interp_to: pd.Series, timeline_interp_from: pd.Series,
                 value_interp_from: [pd.Series, pd.DataFrame], **kwargs):

        self._value_interpolator = {}

        self.timeline_interp_to = timeline_interp_to
        self.timeline_interp_from = timeline_interp_from
        self.value_interp_from = value_interp_from

        config = deepcopy(self._DEFAULT_INTERP_CONFIG)
        config.update(kwargs)
        self._timeline_interpolator = interp1d(self.timeline_interp_to, self.timeline_interp_from, **config)

    def add_value_to_interp(self, name, **kwargs):
        if name in self._value_interpolator.keys():
            prep_logger.info(f"[{self._name}] Overwriting interpolator: {name}")
        else:
            prep_logger.info(f"[{self._name}] Creating interpolator: {name}")
        interp_time = self._timeline_interpolator(self.value_interp_from.index.astype(float).to_numpy())
        config = deepcopy(self._DEFAULT_INTERP_CONFIG)
        config.update(kwargs)
        self._value_interpolator[name] = interp1d(interp_time,
                                                  self.value_interp_from[name].astype(np.float32).to_numpy(), **config)
        return self._value_interpolator[name]

    def interp(self, name, timepoint_interp_to, **kwargs):
        if name in self.value_interp_from.columns:
            interpolator = self.add_value_to_interp(name, **kwargs)
        else:
            prep_logger.error(f'[{self._name}] Unknown interpolator: {name}')
        return interpolator(timepoint_interp_to)


class TimelineAlignment:
    _name = 'TimelineAlignment'

    def __init__(self, ca_timeline_fn: str, behav_timeline_fn: str,
                 comm_channel_name=' GPIO-3'):

        self.ca_timeline = None
        self.behav_timeline = None
        self.behav_data = None

        self._TTL_received_ca_timeline = None
        self._TTL_emitted_behav_timeline = None
        self._ready_to_build_interpolator = [False, False]
        self._interpolator = None

        try:
            self.load_ca_timeline(ca_timeline_fn, comm_channel_name=comm_channel_name)
        except Exception:
            prep_logger.error(traceback.format_exc())

        try:
            self.load_behav_timeline(behav_timeline_fn)
        except Exception:
            prep_logger.error(traceback.format_exc())

        if not self._ready_to_build_interpolator:
            prep_logger.error(
                f'[{self._name}] Errors in loading calcium and/or behavior timeline, check if the file names and parameters are correct before further processing')

    def load_ca_timeline(self, ca_fn, comm_channel_name=' GPIO-3'):
        '''
        Load Ca timline Excel file into the attributes: ca_timeline and ca_timeline_table
        :param ca_fn: calcium timeline file name (only support xlsx, xls, csv files)
        :param comm_channel_name: the GPIO channel name (e.g. GPIO-1)
        # :return: None
        '''
        prep_logger.info(f'[{self._name}] Importing Ca timeline file...')
        comm_channel_column = ' Channel Name'
        extension_name = ca_fn.split('.')[-1]
        file_postfix = ca_fn.strip(' ').split('_')[-1][:-(len(extension_name) + 1)].lower()
        expected_postfix = "gpio"
        if expected_postfix in file_postfix:
            if len(file_postfix) > len(expected_postfix):
                comm_channel_name = f" GPIO-{file_postfix[len(expected_postfix):]}"
            self.ca_timeline = csv_loader(ca_fn, index_col=0)
            self.ca_timeline = self.ca_timeline.sort_index()
            self.ca_timeline = self.ca_timeline.iloc[
                (self.ca_timeline[comm_channel_column] == comm_channel_name).to_numpy(), -1]
        else:
           prep_logger.error(
                f'\n[{self._name}] Incorrect file postfix: the file name should be end with "{expected_postfix}", '
                f'instead it is {file_postfix}')
        self._ready_to_build_interpolator[0] = True
        prep_logger.info(f'[{self._name}] Ca timeline file imported.')

    def load_behav_timeline(self, behav_fn):
        '''
        Load Behavior timline Excel file into the attributes: behav_timeline and behav_timeline_table
        :param ca_fn: calcium timeline file name (only support xlsx, xls, csv files)
        :param comm_channel_column: the TTL communicating index column name in the Excel sheet
        :param comm_channel_name: the GPIO channel name (e.g. GPIO-1)
        :param TTL_value_col: the TTL value column name in the Excel sheet
        :return: None
        '''
        prep_logger.info(f'[{self._name}] Importing behavior timeline file...')
        self._behav_fn = behav_fn
        extension_name = behav_fn.split('.')[-1]
        file_postfix = behav_fn.strip(' ').split('_')[-1][:-(len(extension_name) + 1)].lower()
        expected_postfix = ["behavior", "fc"]
        if file_postfix in expected_postfix:
            if file_postfix == 'behavior':
                self.behav_timeline, self.behav_data = self._load_regular_behav_data(behav_fn)
            else:
                self.behav_timeline, self.behav_data = self._load_fc_behav_data(behav_fn)
        else:
           prep_logger.error(
                f'\n[{self._name}] Incorrect file postfix: the file name should be end with "{expected_postfix}", '
                f'instead it is {file_postfix}')
        self._ready_to_build_interpolator[1] = True
        prep_logger.info(f'[{self._name}] Behavior timeline file imported.')

    def _load_regular_behav_data(self, behav_fn):
        behav_timeline = csv_loader(behav_fn, sheet_name='Hardware-Arena 1', header=36, index_col=0).iloc[1:, ]
        behav_timeline = behav_timeline.iloc[
            (behav_timeline['Name'] == 'Is output 1 High').to_numpy(), -1]
        behav_timeline = behav_timeline.sort_index()

        behav_data = csv_loader(behav_fn, sheet_name='Track-Arena 1-Subject 1', header=36, index_col=0).iloc[
                     1:, ]
        behav_data = behav_data.replace('-', np.nan).astype(np.float32)
        return behav_timeline, behav_data

    def _load_fc_behav_data(self, fc_fn):
        behav_datasheet = csv_loader(fc_fn, index_col=0)
        inscopix_channel_name = next(i for i in behav_datasheet.columns if 'inscopix active' in i.lower())
        behav_timeline = behav_datasheet[inscopix_channel_name]
        behav_timeline = behav_timeline.sort_index()
        behav_timeline = behav_timeline.diff() > 0
        behav_timeline = behav_timeline.astype(int)

        behav_data = behav_datasheet.astype(np.float32)
        return behav_timeline, behav_data

    def detect_ca_TTL_emission_onsets(self, TTL_BINARIZE_THRESHOLD):
        if not all(self._ready_to_build_interpolator):
           prep_logger.error(
                f'[{self._name}] Errors in loading calcium and/or behavior timeline, check if the file names and '
                f'parameters are correct before further processing')
            # Compute TTL emission time in behavior timeline
        prep_logger.info(f'[{self._name}] Compute TTL emission time in [Behavior Timeline]...')
        behav_emission_index = np.where(self.behav_timeline)[0]
        self._TTL_emitted_behav_timeline = self.behav_timeline.iloc[behav_emission_index]
        prep_logger.info(f' done. {behav_emission_index.shape[0]} events emitted')

        # Compute TTL received time in calcium GPIO timeline
        prep_logger.info(f'[{self._name}] Compute TTL received time in [Ca Timeline]...')
        binarized_signal = (self.ca_timeline > TTL_BINARIZE_THRESHOLD).astype(int)
        on_edge_index = np.where(binarized_signal.diff() == 1)[0]
        self._TTL_received_ca_timeline = self.ca_timeline.iloc[on_edge_index]  #
        prep_logger.info(f' done. {on_edge_index.shape[0]} events emitted')


    def build_interpolator(self, TTL_BINARIZE_THRESHOLD=400, interp_direction='ca2behav', **kwargs):
        '''
        Detect TTL events in the Ca Timeline and behavior timeline, and use these events to construct the interpolator
        :param TTL_BINARIZE_THRESHOLD: threshold for detecting TTL events in calicum timeline, if set to None then skip the detection steps.
        :param interp_direction: 'ca2behav': interpolate calcium time into behavior time; 'behav2ca': interpolate behavior time into calcium time
        :param kwargs: other parameters to pass to the scipy interp1d function
        :return: timeline_interpolator: interpolator for converting one timeline to the other
        '''
        if not all(self._ready_to_build_interpolator):
           prep_logger.error(
                f'[{self._name}] Errors in loading calcium and/or behavior timeline, check if the file names and parameters are correct before further processing')

        if TTL_BINARIZE_THRESHOLD is not None:
            self.detect_ca_TTL_emission_onsets(TTL_BINARIZE_THRESHOLD)

        if self._TTL_emitted_behav_timeline is None or self._TTL_received_ca_timeline is None:
            prep_logger.error('TTL emission and/or received event undetected.\n'
                            'Call function "detect_ca_TTL_emission_onsets" before building interpolator')

        behav_event_num = self._TTL_emitted_behav_timeline.shape[0]
        ca_event_num = self._TTL_received_ca_timeline.shape[0]
        ready_to_interpolate = False
        if behav_event_num > ca_event_num:
            # If the TTL events detected in the two timelines are not the same, then consider changing the
            # TTL_BINARIZE_THRESHOLD, if still does not work, then use the function 'visua_debugger' to determine
            # which events in the Ca Timeline are missing.
            prep_logger.error(f'Received fewer Ca TTL events ({ca_event_num}) than the number of behavior '
                            f'TTL events emitted({behav_event_num}).')
        elif behav_event_num < ca_event_num:
            trimmed_ca_timeline = self.TTL_received_ca_timeline[:behav_event_num]
            self.transmission_gap_time = self.TTL_emitted_behav_timeline - trimmed_ca_timeline  # compiled the time differences in all TTL emission-receiving event pairs
            self.start_time_difference = self.transmission_gap_time[
                0]  # the start time difference is defined as the time difference in first TTL event pair
            self.transmission_variation = np.std(
                self.transmission_gap_time)  # the time difference variation is defined as the standard deviation of all TTL event pairs
            extra_behav_entry = self._TTL_received_ca_timeline.iloc[
                (self._TTL_received_ca_timeline.index+self.start_time_difference+self.transmission_variation) > self._TTL_emitted_behav_timeline.index.max()]
            if extra_behav_entry.shape[0] != (ca_event_num-behav_event_num):
                prep_logger.warn(f'Trimmed ({(ca_event_num-behav_event_num)}) events, however there are'
                              f' {extra_behav_entry.shape[0]} TTL events emitted after the behavior timeline finished.')
            self._TTL_received_ca_timeline = self._TTL_received_ca_timeline.iloc[:behav_event_num]
            ready_to_interpolate = True
        else:
            self.transmission_gap_time = self.TTL_emitted_behav_timeline - self.TTL_received_ca_timeline  # compiled the time differences in all TTL emission-receiving event pairs
            self.start_time_difference = self.transmission_gap_time[
                0]  # the start time difference is defined as the time difference in first TTL event pair
            self.transmission_variation = np.std(
                self.transmission_gap_time)  # the time difference variation is defined as the standard deviation of all TTL event pairs
            ready_to_interpolate = True

        if ready_to_interpolate:
            if interp_direction == 'ca2behav':
                timeline_interpolator = TimeInterpolator(self.TTL_emitted_behav_timeline, self.TTL_received_ca_timeline,
                                                         self.behav_data,
                                                         **kwargs)  # build a scipy linear interpolator to convert the times of the events in one timeline to another
            elif interp_direction == 'behav2ca':
                timeline_interpolator = TimeInterpolator(self.TTL_received_ca_timeline, self.TTL_emitted_behav_timeline,
                                                         self.behav_data,
                                                         **kwargs)  # build a scipy linear interpolator to convert the times of the events in one timeline to another
            else:
                prep_logger.error(f'[{self._name}] Invalid interpolation direction: {interp_direction}')
                return None
            prep_logger.info(f'[{self._name}] Interpolator built.\n' +
                  ' ' * (
                          len(self._name) + 3) + f'Start time difference: {self.start_time_difference * 1000: .2f}ms.\n' +
                  ' ' * (
                          len(self._name) + 3) + f'Transmission time variation (std): {self.transmission_variation * 1000: .2f}ms.')

            self._interpolator = timeline_interpolator

            return timeline_interpolator

    def resample_behav_measure(self, time_pnt: np.ndarray):
        if self._interpolator is None:
            prep_logger.error(f'[{self._name}] No interpolator built. Run build_interpolator() first')
        assert type(
            time_pnt) is np.ndarray, f'[{self._name}] Invalid type of time data (expecting numpy ndarray), instead it is {type(time_pnt)}'
        resampled_dict = {}
        for name in self._interpolator.value_interp_from.columns:
            resampled_dict[name] = self._interpolator.interp(name, time_pnt).astype(np.float32)
        resampled_dict = pd.DataFrame(resampled_dict, index=time_pnt)
        prep_logger.info(
            f'The following measures have been interpolated: \n{[i for i in self._interpolator.value_interp_from.columns]}')
        return resampled_dict

    @property
    def TTL_emitted_behav_timeline(self):
        return self._TTL_emitted_behav_timeline.index.to_numpy()

    @property
    def TTL_received_ca_timeline(self):
        return self._TTL_received_ca_timeline.index.to_numpy()

    def visual_debugger(self):
        plt.figure('Ca timeline TTL onset')
        plt.plot(self.ca_timeline.index.to_numpy(), self.ca_timeline.to_numpy(), 'k')
        plt.scatter(self._TTL_received_ca_timeline.index.to_numpy(), self._TTL_received_ca_timeline.to_numpy(), c='r',
                    marker='*')


def batch_interp(interpolator: TimeInterpolator, name: [list, str], time_interp_to):
    '''
    Batch interpolation using prebuilt time interpolators

    :param interpolator: A TimeInterpolator object returned from TimelineAlignment.build_interpolator function
    :param name: a list of the column name (str) of the dataframe to be interpolated, if name == 'all', then interpolate all columns
    :param time_interp_to: the target time points interpolated to
    :param dataframe_interp_from: a pandas.DataFrame object whose index is the time to be interpolated, and other columns are the values to be interpolated
    :return:
        interp_val_dict: a dictionary whose keys are the selected column names and values are the interpolated values
    '''
    interp_val_dict = {}
    if type(name) is str:
        if name == 'all':
            name = list(interpolator.value_interp_from.columns)
        else:
            prep_logger.error(ValueError(f'Name should be either a list or "all" but not {name}'))
    for n in name:
        if n not in interpolator.value_interp_from.columns:
            prep_logger.warn(f"[batch_interp] The input dataframe does not have column name {n}.")
        else:
            interp_val_dict[n] = interpolator.interp(n, time_interp_to)
    return interp_val_dict


def align_ca_behav_data(data_path, ID, comm_channel_name=' GPIO-3', TTL_BINARIZE_THRESHOLD=40000,
                        scipy_interp_setting={}, use_independent_logger=False):
    _name = 'align_ca_behav_data'
    if use_independent_logger:
        create_logger(data_path)
    behav_tl_fn, ca_tl_fn, ca_data_fn = exp_folder_loader(data_path)
    ta = TimelineAlignment(ca_tl_fn, behav_tl_fn, comm_channel_name)  # create timeline alignment object
    ta.build_interpolator(TTL_BINARIZE_THRESHOLD)  # build interpolator with the timeline alignment object
    ca_data, cell_accepted = load_ca_data(ca_data_fn)
    resampled = ta.resample_behav_measure(ca_data.index.to_numpy())
    h5_fn = f'{data_path}/preprocessed_{ID}.h5'
    if os.path.exists(h5_fn):
        prep_logger.error('Preprocessed file already exist.')

    with h5.File(h5_fn, 'w') as hf:
        hf.attrs['folder_path'] = data_path

        ca_group = hf.create_group('Raw Ca data')
        ca_group.attrs['ca_data_path'] = ca_data_fn
        index_col = ca_data.index.to_numpy().astype(float)
        tmp = ca_group.create_dataset('time', index_col.shape, index_col.dtype, data=index_col)
        tmp.attrs['shape'] = index_col.shape
        for i in ca_data.columns:
            data = ca_data[i].astype(float)
            tmp = ca_group.create_dataset(i, data.shape, data.dtype, data=data)
            tmp.attrs['shape'] = data.shape
            if i in cell_accepted:
                tmp.attrs['accepted'] = True
            else:
                tmp.attrs['accepted'] = False

        behav_group = hf.create_group('Raw behavior data')
        behav_group.attrs['behav_data_path'] = behav_tl_fn
        index_col = ta.behav_data.index.to_numpy().astype(float)
        tmp = behav_group.create_dataset('time', index_col.shape, index_col.dtype, data=index_col)
        tmp.attrs['shape'] = index_col.shape
        for i in ta.behav_data.columns:
            data = ta.behav_data[i].astype(float)
            tmp = behav_group.create_dataset(i, data.shape, data.dtype, data=data)
            tmp.attrs['shape'] = data.shape

        timeline_group = hf.create_group('Raw timeline data')
        timeline_group.attrs['ca_timeline_path'] = ca_tl_fn
        timeline_group.attrs['behavior_timeline_path'] = behav_tl_fn
        ca_tl_idx = ta.ca_timeline.index.to_numpy()
        ca_tl_val = ta.ca_timeline.to_numpy()
        tmp = timeline_group.create_dataset('Ca time', ca_tl_idx.shape, ca_tl_idx.dtype, data=ca_tl_idx)
        tmp.attrs['shape'] = ca_tl_idx.shape
        tmp = timeline_group.create_dataset('Ca TTL value', ca_tl_val.shape, ca_tl_val.dtype, data=ca_tl_val)
        tmp.attrs['shape'] = ca_tl_val.shape
        behav_tl_idx = ta.behav_timeline.index.to_numpy().astype(float)
        behav_tl_val = ta.behav_timeline.to_numpy().astype(float)
        tmp = timeline_group.create_dataset('Behavior time', behav_tl_idx.shape, behav_tl_idx.dtype, data=behav_tl_idx)
        tmp.attrs['shape'] = behav_tl_idx.shape
        tmp = timeline_group.create_dataset('Behavior TTL value', behav_tl_val.shape, behav_tl_val.dtype,
                                            data=behav_tl_val)
        tmp.attrs['shape'] = behav_tl_val.shape

        aligned = hf.create_group('Aligned behavior data')
        aligned.attrs['comm_channel_name'] = comm_channel_name
        aligned.attrs['TTL_BINARIZE_THRESHOLD'] = TTL_BINARIZE_THRESHOLD

        for k, v in scipy_interp_setting.items():
            aligned.attrs[k] = v

        index_col = resampled.index.to_numpy()
        aligned.create_dataset('time', index_col.shape, index_col.dtype, data=index_col)
        for i in resampled.columns:
            data = resampled[i].astype(float)
            tmp = aligned.create_dataset(i, data.shape, data.dtype, data=data)
            tmp.attrs['shape'] = data.shape

    prep_logger.info(f'{_name} Completed, saved as: {data_path}/preprocessed_{ID}.h5')
