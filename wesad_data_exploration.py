import concurrent.futures
from datetime import timedelta

import gzip
import logging
import matplotlib as plt
import neurokit2 as nk
import numpy as np
import os
import pandas as pd
import pyhrv
import scipy.signal as scisig
import scipy.stats
import shutil
import time
from urllib.request import Request, urlopen
import zipfile

import cvxEDA

#matplotlib inline
plt.rcParams['figure.figsize'] = [10, 7]  # Bigger images

class WesadDataLoader():
    """Downloads and load data from the WESAD dataset
        
        Source URI: https://uni-siegen.sciebo.de/s/pYjSgfOVs6Ntahr/download
    """
    
    LABEL = 'label'
    SIGNAL = 'signal'
    SUBJECT = 'subject'
    
    WRIST_DEV = 'wrist'
    CHEST_DEV = 'chest'
    
    DATASET_NAME = 'WESAD'
    DATASET_URI = 'https://uni-siegen.sciebo.de/s/pYjSgfOVs6Ntahr/download'
    
    def __init__(self, subject, basepath='.'):
        self.logger = logging.getLogger(WesadDataLoader.__name__)
        self.logger.info('Init...')
        self.chest_modalities = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
        self.wrist_modalities = ['ACC', 'BVP', 'EDA', 'TEMP']
        self.mod_samp_rate = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'chest': 700}  # Hz
        WesadDataLoader.download(basepath)
        basepath = os.path.join(os.path.abspath(basepath), WesadDataLoader.DATASET_NAME, subject)
        if not os.path.isdir(basepath):
            raise Exception(f'Dataset path does not exist or is not a directory: {basepath}')
        data_file = os.path.join(basepath, f'{subject}.pkl')
        if not os.path.exists(data_file):
            raise Exception(f'Data file does not exists: {data_file}')
#         with open(subject + '.pkl', 'rb') as file:
#             data = pickle.load(file, encoding='latin1')
        self.data = pd.read_pickle(data_file)
    
    @staticmethod
    def download(basepath):
        filename = os.path.join(os.path.abspath(basepath), f'{WesadDataLoader.DATASET_NAME}.zip')
        data_folder = os.path.join(os.path.abspath(basepath), WesadDataLoader.DATASET_NAME)
        if not os.path.isdir(data_folder) and not os.path.exists(filename):
            print('Downloading dataset...')
            start = time.time()
            response = urlopen(WesadDataLoader.DATASET_URI)
            print(f'Elapsed: {time.time() - start} secs')
        if not os.path.isdir(data_folder):
            with open(filename, 'wb') as out_file:
                print('Saving dataset locally...')
                start = time.time()
                shutil.copyfileobj(response, out_file)
            out_file.close()
            print(f'Elapsed: {time.time() - start} secs')
            start = time.time()
            while not zipfile.is_zipfile(filename):
                print('Wait..')
            print('Found Zip...')
            print(f'Elapsed: {time.time() - start} secs')
            with zipfile.ZipFile(filename) as zf:
                print('Extracting files...')
                start = time.time()
                zf.extractall()
            print(f'Elapsed: {time.time() - start} secs')
            print('Done!')

    def get_labels(self):
        return self.data[WesadDataLoader.LABEL]

    def get_wrist_data(self):
        """"""
        #label = self.data[self.keys[0]]
#         assert subject == self.data[self.keys[1]]
        signal = self.data[WesadDataLoader.SIGNAL]
        wrist_data = signal[WesadDataLoader.WRIST_DEV]
        # Adding Resp modality from chest device
        wrist_data.update({'Resp': self.data[WesadDataLoader.SIGNAL][WesadDataLoader.CHEST_DEV]['Resp']})
        return wrist_data

    def get_chest_data(self):
        """"""
        signal = self.data[WesadDataLoader.SIGNAL]
        chest_data = signal[WesadDataLoader.CHEST_DEV]
        return chest_data
    
#time
BASE_PATH = './'
# WesadDataLoader.download('.')
DATASET_PATH = os.path.join(BASE_PATH, WesadDataLoader.DATASET_NAME)
subjects = [dir_ for dir_ in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, dir_))]
# subjects = ['S3']
obj_data = {}

for subject in subjects:
    obj_data[subject] = WesadDataLoader(subject=subject, basepath=BASE_PATH)
    
# Checking dataset size
sampling_rate=700
window_size=60
window_shift=0.25

baseline_rec = 0
stress_rec = 0
amusement_rec = 0
total_segmented = 0
print('Subjects', obj_data.keys())
for sub in obj_data.keys():
    data = obj_data[sub].get_chest_data()
    labels = obj_data[sub].get_labels()
    baseline = np.asarray([idx for idx,val in enumerate(labels) if val == 1])
    stress = np.asarray([idx for idx,val in enumerate(labels) if val == 2])
    amusement = np.asarray([idx for idx,val in enumerate(labels) if val == 3])

    baseline_rec += baseline.shape[0]
    stress_rec += stress.shape[0]
    amusement_rec += amusement.shape[0]
    conditions = [baseline, stress, amusement]
    
    subtotal = 0
    for cond in conditions:
        subtotal += len(list(range(0, data['ACC'][cond].shape[0] - (sampling_rate * window_size), int(sampling_rate * window_shift))))
    print('Subject', sub, subtotal)
    total_segmented += subtotal

(baseline_rec + stress_rec + amusement_rec), total_segmented

def get_slope(series):
    linreg = scipy.stats.linregress(np.arange(len(series)), series )
    slope = linreg[0]
    return slope


def get_freq_features(series):
    # Peak frequency is simply the frequency of maximum power
    f, Pxx = scisig.periodogram(series) # Estimate power spectral density (PSD) using a periodogram
    psd_dict = {amp: freq for amp, freq in zip(Pxx, f)}
    peak_freq = psd_dict[max(psd_dict.keys())]
    # Mean freq: http://luscinia.sourceforge.net/page26/page35/page35.html
    mean_freq = np.dot(Pxx, f) / np.sum(Pxx)
    avg_power = np.sum(Pxx) / 2
    # Median freq: http://luscinia.sourceforge.net/page26/page36/page36.html
    median_freq = f[(np.cumsum(f) > avg_power).argmax()]

    return peak_freq, mean_freq, median_freq

def compute_features(data, condition, sampling_rate=700, window_size=60, window_shift=0.25):

    index = 0
    init = time.time()

    # data cleaning
    ## ECG
    ecg_cleaned = nk.ecg_clean(data["ECG"][condition].flatten(), sampling_rate=sampling_rate)
    ## == OLD
    # ecg_rpeaks, _ = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
    # ecg_hr = nk.signal_rate(ecg_rpeaks, sampling_rate=sampling_rate)
    ## ==
    ## EDA
    ## 5Hz lowpass filter
    eda_highcut = 5
    eda_filtered = nk.signal_filter(data['EDA'][condition].flatten(), sampling_rate=sampling_rate, highcut=eda_highcut)
    eda_cleaned = nk.standardize(eda_filtered)
    # TODO: not sure about the approach. cvxeda takes longer periods
    # phasic_tonic = nk.eda_phasic(cleaned, sampling_rate=700, method='cvxeda')
    eda_phasic_tonic = nk.eda_phasic(eda_cleaned, sampling_rate=sampling_rate)
    eda_phasic_tonic['t'] = [(1 / sampling_rate) * i for i in range(eda_phasic_tonic.shape[0])]
    eda_scr_peaks, scr_info = nk.eda_peaks(eda_phasic_tonic['EDA_Phasic'], sampling_rate=sampling_rate)
    ## EMG
    ## For 5 sec window signal
    ## More on DC Bias https://www.c-motion.com/v3dwiki/index.php/EMG:_Removing_DC_Bias
    emg_lowcut = 50
    emg_filtered_dc = nk.signal_filter(data['EMG'][condition].flatten(), sampling_rate=sampling_rate, lowcut=emg_lowcut)
    # OR 100 Hz highpass Butterworth filter followed by a constant detrending
    # filtered_dc = nk.emg_clean(chest_data_dict['EMG'][baseline].flatten(), sampling_rate=700)
    ## For 60 sec window signal
    # 50Hz lowpass filter
    emg_highcut = 50
    emg_filtered = nk.signal_filter(data['EMG'][condition].flatten(), sampling_rate=sampling_rate, highcut=emg_highcut)
    ## Resp
    ## Method biosppy important to appply bandpass filter 0.1 - 0.35 Hz
    resp_processed, _ = nk.rsp_process(data['Resp'][condition].flatten(), sampling_rate=sampling_rate, method='biosppy')

    print('Elapsed Preprocess', str(timedelta(seconds=time.time() - init)))
    init = time.time()

    chest_df_5 = pd.DataFrame() # For 5 sec window size
    chest_df = pd.DataFrame()

    window = int(sampling_rate * window_size)
    for i in range(0, data['ACC'][condition].shape[0] - window, int(sampling_rate * window_shift)):

        # ACC
        w_acc_data = data['ACC'][condition][i: window + i]
        acc_x_mean, acc_y_mean, acc_z_mean = np.mean(w_acc_data, axis=0)  # Feature
        acc_x_std, acc_y_std, acc_z_std = np.std(w_acc_data, axis=0)  # Feature
        acc_x_peak, acc_y_peak, acc_z_peak = np.amax(w_acc_data, axis=0)  # Feature
        acc_x_absint, acc_y_absint, acc_z_absint = np.abs(np.trapz(w_acc_data, axis=0))  # Feature
        xyz = np.sum(w_acc_data, axis=0)
        xyz_mean = np.mean(xyz)  # Feature
        xyz_std = np.std(xyz)  # Feature
        xyz_absint = np.abs(np.trapz(xyz))  # Feature


        # == OLD
        # ## ECG
        # w_ecg_rpeaks = ecg_rpeaks[i: window + i]
        # # HR
        # w_ecg_hr = ecg_hr[i: window + i]
        # hr_mean = np.mean(w_ecg_hr)  # Feature
        # hr_std = np.std(w_ecg_hr)  # Feature
        # # HRV Time-domain Indices
        # # HRV_MeanNN
        # # HRV_SDNN
        # # HRV_pNN50
        # # HRV_RMSSD -> Root mean square of the HRV
        # # HRV_HTI -> Triangular interpolation index
        # hrv_time = nk.hrv_time(w_ecg_rpeaks, sampling_rate=sampling_rate, show=False)
        # hrv_mean = hrv_time.loc[0, 'HRV_MeanNN']  # Feature
        # hrv_std = hrv_time.loc[0, 'HRV_SDNN']  # Feature
        # # TODO: NN50
        # # hrv_NN50 = 
        # hrv_pNN50 = hrv_time.loc[0, 'HRV_pNN50']  # Feature
        # hrv_TINN = hrv_time.loc[0, 'HRV_HTI']  # Feature
        # hrv_rms = hrv_time.loc[0, 'HRV_RMSSD']  # Feature

        # # HRV Frequency-domain Indices
        # # TODO: get NaN values within windows (*)
        # # HRV_ULF *
        # # HRV_LF *
        # # HRV_HF 
        # # HRV_VHF
        # # HRV_LFHF - Ratio LF/HF *
        # # HRV_LFn *
        # # HRV_HFn
        # hrv_freq = nk.hrv_frequency(w_ecg_rpeaks, sampling_rate=sampling_rate, ulf=(0.01, 0.04), lf=(0.04, 0.15), hf=(0.15, 0.4), vhf=(0.4, 1.))
        # hrv_ULF = hrv_freq.loc[0, 'HRV_ULF']  # Feature
        # hrv_LF = hrv_freq.loc[0, 'HRV_LF']  # Feature
        # hrv_HF = hrv_freq.loc[0, 'HRV_HF']  # Feature
        # hrv_VHF = hrv_freq.loc[0, 'HRV_VHF']  # Feature
        # hrv_lf_hf_ratio = hrv_freq.loc[0, 'HRV_LFHF']  # Feature
        # hrv_f_sum = np.nansum(np.hstack((hrv_ULF, hrv_LF, hrv_HF, hrv_VHF)))
        # # TODO: rel_f
        # # hrv_rel_f = 
        # hrv_LFn = hrv_freq.loc[0, 'HRV_LFn']  # Feature
        # hrv_HFn = hrv_freq.loc[0, 'HRV_HFn']  # Feature
        # ==

        ## ECG 
        w_ecg_cleaned = ecg_cleaned[i: window + i]
        _, ecg_info = nk.ecg_peaks(w_ecg_cleaned, sampling_rate=sampling_rate)
        w_ecg_rpeaks = ecg_info['ECG_R_Peaks']
        ecg_nni = pyhrv.tools.nn_intervals(w_ecg_rpeaks)
        # HR
        rs_hr = pyhrv.time_domain.hr_parameters(ecg_nni)
        hr_mean = rs_hr['hr_mean']  # Feature
        hr_std = rs_hr['hr_std']  # Feature
        # HRV-time
        rs_hrv = pyhrv.time_domain.nni_parameters(ecg_nni)
        hrv_mean = rs_hrv['nni_mean']  # Feature
        hrv_std = pyhrv.time_domain.sdnn(ecg_nni)['sdnn']  # Feature
        rs_nn50 = pyhrv.time_domain.nn50(ecg_nni)
        hrv_NN50 = rs_nn50['nn50']  # Feature
        hrv_pNN50 = rs_nn50['pnn50']  # Feature
        hrv_time = nk.hrv_time(w_ecg_rpeaks, sampling_rate=sampling_rate, show=False)
        hrv_TINN = hrv_time.loc[0, 'HRV_TINN']  # Feature
        hrv_rms = pyhrv.time_domain.rmssd(ecg_nni)['rmssd']  # Feature
        # HRV-freq
        hrv_freq = pyhrv.frequency_domain.welch_psd(ecg_nni, fbands={'ulf': (0.01, 0.04), 'vlf': (0.04, 0.15), 'lf': (0.15, 0.4), 'hf': (0.4, 1)}, mode='dev')
        # hrv_freq = hrv_freq.as_dict()
        hrv_freq = hrv_freq[0]
        hrv_ULF = hrv_freq['fft_abs'][0]  # Feature
        hrv_LF = hrv_freq['fft_abs'][1]  # Feature
        hrv_HF = hrv_freq['fft_abs'][2]  # Feature
        hrv_VHF = hrv_freq['fft_abs'][3]  # Feature
        hrv_lf_hf_ratio = hrv_freq['fft_ratio']  # Feature
        hrv_f_sum = hrv_freq['fft_total']  # Feature
        hrv_rel_ULF = hrv_freq['fft_rel'][0]  # Feature
        hrv_rel_LF = hrv_freq['fft_rel'][1]  # Feature
        hrv_rel_HF = hrv_freq['fft_rel'][2]  # Feature
        hrv_rel_VHF = hrv_freq['fft_rel'][3]  # Feature
        hrv_LFn = hrv_freq['fft_norm'][0]  # Feature
        hrv_HFn = hrv_freq['fft_norm'][1]  # Feature

        # EDA
        w_eda_data = eda_cleaned[i: window + i]
        w_eda_phasic_tonic = eda_phasic_tonic[i: window + i]

        eda_mean = np.mean(w_eda_data)  # Feature
        eda_std = np.std(w_eda_data)  # Feature
        eda_min = np.amin(w_eda_data)  # Feature
        eda_max = np.amax(w_eda_data)  # Feature
        # dynamic range: https://en.wikipedia.org/wiki/Dynamic_range
        eda_slope = get_slope(w_eda_data)  # Feature
        eda_drange = eda_max / eda_min  # Feature
        eda_scl_mean = np.mean(w_eda_phasic_tonic['EDA_Tonic'])  # Feature
        eda_scl_std = np.std(w_eda_phasic_tonic['EDA_Tonic'])  # Feature
        eda_scr_mean = np.mean(w_eda_phasic_tonic['EDA_Phasic'])  # Feature
        eda_scr_std = np.std(w_eda_phasic_tonic['EDA_Phasic'])  # Feature
        eda_corr_scl_t = nk.cor(w_eda_phasic_tonic['EDA_Tonic'], w_eda_phasic_tonic['t'], show=False)  # Feature
        
        eda_scr_no = eda_scr_peaks['SCR_Peaks'][i: window + i].sum()  # Feature
        # Sum amplitudes in SCR signal
        ampl = scr_info['SCR_Amplitude'][i: window + i]
        eda_ampl_sum = np.sum(ampl[~np.isnan(ampl)])  # Feature
        # TODO: 
        # eda_t_sum = 

        scr_peaks, scr_properties = scisig.find_peaks(w_eda_phasic_tonic['EDA_Phasic'], height=0)
        width_scr = scisig.peak_widths(w_eda_phasic_tonic['EDA_Phasic'], scr_peaks, rel_height=0)
        ht_scr = scr_properties['peak_heights']
        eda_scr_area = 0.5 * np.matmul(ht_scr, width_scr[1])  # Feature

        # EMG
        ## 5sec
        w_emg_data = emg_filtered_dc[i: window + i]
        emg_mean = np.mean(w_emg_data)  # Feature
        emg_std = np.std(w_emg_data)  # Feature
        emg_min = np.amin(w_emg_data)
        emg_max = np.amax(w_emg_data)
        emg_drange = emg_max / emg_min  # Feature
        emg_absint = np.abs(np.trapz(w_emg_data))  # Feature
        emg_median = np.median(w_emg_data)  # Feature
        emg_perc_10 = np.percentile(w_emg_data, 10)  # Feature
        emg_perc_90 = np.percentile(w_emg_data, 90)  # Feature
        emg_peak_freq, emg_mean_freq, emg_median_freq = get_freq_features(w_emg_data)  # Features
        # TODO: PSD -> energy in seven bands
        # emg_psd = 

        ## 60 sec
        peaks, properties = scisig.find_peaks(emg_filtered[i: window + i], height=0)
        emg_peak_no = peaks.shape[0]
        emg_peak_amp_mean = np.mean(properties['peak_heights'])  # Feature
        emg_peak_amp_std = np.std(properties['peak_heights'])  # Feature
        emg_peak_amp_sum = np.sum(properties['peak_heights'])  # Feature
        emg_peak_amp_max = np.abs(np.amax(properties['peak_heights']))
        # https://www.researchgate.net/post/How_Period_Normalization_and_Amplitude_normalization_are_performed_in_ECG_Signal
        emg_peak_amp_norm_sum = np.sum(properties['peak_heights'] / emg_peak_amp_max)  # Feature

        # Resp
        w_resp_data = resp_processed[i: window + i]
        ## Inhalation / Exhalation duration analysis
        idx = np.nan
        count = 0
        duration = dict()
        first = True
        for j in w_resp_data[~w_resp_data['RSP_Phase'].isnull()]['RSP_Phase'].to_numpy():
            if j != idx:
                if first:
                    idx = int(j)
                    duration[1] = []
                    duration [0] = []
                    first = False
                    continue
                # print('New value', j, count)
                duration[idx].append(count)
                idx = int(j)
                count = 0 
            count += 1
        resp_inhal_mean = np.mean(duration[1])  # Feature
        resp_inhal_std = np.std(duration[1])  # Feature
        resp_exhal_mean = np.mean(duration[0])  # Feature
        resp_exhal_std = np.std(duration[0])  # Feature
        resp_inhal_duration = w_resp_data['RSP_Phase'][w_resp_data['RSP_Phase'] == 1].count()
        resp_exhal_duration = w_resp_data['RSP_Phase'][w_resp_data['RSP_Phase'] == 0].count()
        resp_ie_ratio = resp_inhal_duration / resp_exhal_duration  # Feature
        resp_duration = resp_inhal_duration + resp_exhal_duration  # Feature
        resp_stretch = w_resp_data['RSP_Amplitude'].max() - w_resp_data['RSP_Amplitude'].min()  # Feature
        resp_breath_rate = len(duration[1])  # Feature
        ## Volume: area under the curve of the inspiration phase on a respiratory cycle
        resp_peaks, resp_properties = scisig.find_peaks(w_resp_data['RSP_Clean'], height=0)
        resp_width = scisig.peak_widths(w_resp_data['RSP_Clean'], resp_peaks, rel_height=0)
        resp_ht = resp_properties['peak_heights']        
        resp_volume = 0.5 * np.matmul(resp_ht, resp_width[1])  # Feature

        # Temp
        w_temp_data = data['Temp'][condition][i: window + i].flatten()
        temp_mean = np.mean(w_temp_data)  # Feature
        temp_std = np.std(w_temp_data)  # Feature
        temp_min = np.amin(w_temp_data)  # Feature
        temp_max = np.amax(w_temp_data)  # Feature
        temp_drange = temp_max / temp_min  # Feature
        temp_slope = get_slope(w_temp_data.ravel())  # Feature


        # chest_df_5 = chest_df_5.append({
        #     'ACC_x_mean': acc_x_mean, 'ACC_y_mean': acc_y_mean, 'ACC_z_mean': acc_z_mean, 'ACC_xzy_mean': xyz_mean,
        #     'ACC_x_std': acc_x_std, 'ACC_y_std': acc_y_std, 'ACC_z_std': acc_z_std, 'ACC_xyz_std': xyz_std,
        #     'ACC_x_absint': acc_x_absint, 'ACC_y_absint': acc_y_absint, 'ACC_z_absint': acc_z_absint, 'ACC_xyz_absint': xyz_absint,
        #     'ACC_x_peak': acc_x_peak, 'ACC_y_peak': acc_y_peak, 'ACC_z_peak': acc_z_peak,
        #     'EMG_mean': emg_mean, 'EMG_std': emg_std, 'EMG_drange': emg_drange, 'EMG_absint': emg_absint, 'EMG_median': emg_median, 'EMG_perc_10': emg_perc_10,
        #     'EMG_perc_90': emg_perc_90, 'EMG_peak_freq': emg_peak_freq, 'EMG_mean_freq': emg_mean_freq, 'EMG_median_freq': emg_median_freq
        # }, ignore_index=True)

        chest_df = chest_df.append({
            'ACC_x_mean': acc_x_mean, 'ACC_y_mean': acc_y_mean, 'ACC_z_mean': acc_z_mean, 'ACC_xzy_mean': xyz_mean,
            'ACC_x_std': acc_x_std, 'ACC_y_std': acc_y_std, 'ACC_z_std': acc_z_std, 'ACC_xyz_std': xyz_std,
            'ACC_x_absint': acc_x_absint, 'ACC_y_absint': acc_y_absint, 'ACC_z_absint': acc_z_absint, 'ACC_xyz_absint': xyz_absint,
            'ACC_x_peak': acc_x_peak, 'ACC_y_peak': acc_y_peak, 'ACC_z_peak': acc_z_peak,
            'ECG_hr_mean': hr_mean, 'ECG_hr_std': hr_std, 'ECG_hrv_NN50': hrv_NN50, 'ECG_hrv_pNN50': hrv_pNN50, 'ECG_hrv_TINN': hrv_TINN, 'ECG_hrv_RMS': hrv_rms,
            'ECG_hrv_ULF': hrv_ULF, 'ECG_hrv_LF': hrv_LF, 'ECG_hrv_HF': hrv_HF, 'ECG_hrv_VHF': hrv_VHF, 'ECG_hrv_LFHF_ratio': hrv_lf_hf_ratio, 'ECG_hrv_f_sum': hrv_f_sum,
            'ECG_hrv_rel_ULF': hrv_rel_ULF, 'ECG_hrv_rel_LF': hrv_rel_LF, 'ECG_hrv_rel_HF': hrv_rel_HF, 'ECG_hrv_rel_VHF': hrv_rel_VHF, 'ECG_hrv_LFn': hrv_LFn, 'ECG_hrv_HFn': hrv_HFn,
            'EDA_mean': eda_mean, 'EDA_std': eda_std, 'EDA_mean': eda_mean, 'EDA_min': eda_min, 'EDA_max': eda_max, 'EDA_slope': eda_slope,
            'EDA_drange': eda_drange, 'EDA_SCL_mean': eda_scl_mean, 'EDA_SCL_std': eda_scl_mean, 'EDA_SCR_mean': eda_scr_mean, 'EDA_SCR_std': eda_scr_std,
            'EDA_corr_SCL_t': eda_corr_scl_t, 'EDA_SCR_no': eda_scr_no, 'EDA_ampl_sum': eda_ampl_sum, 'EDA_scr_area': eda_scr_area,
            'EMG_mean': emg_mean, 'EMG_std': emg_std, 'EMG_drange': emg_drange, 'EMG_absint': emg_absint, 'EMG_median': emg_median, 'EMG_perc_10': emg_perc_10,
            'EMG_perc_90': emg_perc_90, 'EMG_peak_freq': emg_peak_freq, 'EMG_mean_freq': emg_mean_freq, 'EMG_median_freq': emg_median_freq,
            'EMG_peak_no': emg_peak_no, 'EMG_peak_amp_mean':  emg_peak_amp_mean, 'EMG_peak_amp_std':  emg_peak_amp_std, 'EMG_peak_amp_sum':  emg_peak_amp_sum,
            'EMG_peak_amp_norm_sum':  emg_peak_amp_norm_sum,
            'RESP_inhal_mean': resp_inhal_mean, 'RESP_inhal_std': resp_inhal_std, 'RESP_exhal_mean': resp_exhal_mean, 'RESP_exhal_std': resp_exhal_std,
            'RESP_ie_ratio': resp_ie_ratio, 'RESP_duration': resp_duration, 'RESP_stretch': resp_stretch, 'RESP_breath_rate': resp_breath_rate, 'RESP_volume': resp_volume,
            'TEMP_mean': temp_mean, 'TEMP_std': temp_std, 'TEMP_min': temp_min, 'TEMP_max': temp_max, 'TEMP_drange': temp_drange, 'TEMP_slope': temp_slope
        }, ignore_index=True)


        # index += 1
        # if index % 10 == 0:
        #     break
    
    print('Elapsed Process', condition.shape[0], str(timedelta(seconds=time.time() - init)))
    return chest_df, chest_df_5

def process_subject(subject_data, cond_to_process, max_workers=6):
    rs = dict()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_label = {executor.submit(compute_features, subject_data, cond): label for label, cond in cond_to_process}
        for future in concurrent.futures.as_completed(future_to_label):
            label = future_to_label[future]
            try:
                data, _ = future.result()
                print(label, data.shape)
                rs[label] = data
            except Exception as exc:
                print('%r generated an exception: %s' % (label, exc))
    return rs

subjects = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']
for subject in subjects:
    print('Subject', subject)
    chest_data_dict = obj_data[subject].get_chest_data()
    labels = obj_data[subject].get_labels()
    chest_dict_length = {key: len(value) for key, value in chest_data_dict.items()}
    print(chest_dict_length)

    # Get labels
    baseline = np.asarray([idx for idx,val in enumerate(labels) if val == 1])
    stress = np.asarray([idx for idx,val in enumerate(labels) if val == 2])
    amusement = np.asarray([idx for idx,val in enumerate(labels) if val == 3])

    print("Baseline:", chest_data_dict['ECG'][baseline].shape)
    print("Stress:", chest_data_dict['ECG'][stress].shape)
    print("Amusement:", chest_data_dict['ECG'][amusement].shape)

    # Process Subject
    to_process = zip(['baseline', 'stress', 'amusement'], [baseline, stress, amusement])
    # to_process = zip(['baseline'], [baseline])
    #time 
    subject_data = process_subject(chest_data_dict, cond_to_process=to_process)
    ## Labeling
    subject_data['baseline']['label'] = 1
    subject_data['baseline']['subject'] = subject
    subject_data['stress']['label'] = 2
    subject_data['stress']['subject'] = subject
    subject_data['amusement']['label'] = 3
    subject_data['amusement']['subject'] = subject
    ## Storing
    dfs = [v for k, v in subject_data.items()]
    df_subject = pd.concat(dfs)
    print('Generated dataset for', subject, df_subject.shape)
    df_subject.head()
    df_subject.reset_index().to_feather(f'{subject}.feather')
    
