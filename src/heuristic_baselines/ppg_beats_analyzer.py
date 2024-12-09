import numpy as np
import pandas as pd
from pathlib import Path
import scipy

winsize = 8 # 8 seconds
outdict = {'part':[], 'body_loc':[], 'MAE':[], 'MAPE':[], 'method':[], 'coverage':[], 'RMSE':[], 'corr':[], 'dataset':[]}

#%%
def load_wildppg_participant(path):
    """
    Loads the data of a WildPPG participant and cleans it to receive nested dictionaries
    """
    loaded_data = scipy.io.loadmat(path)
    loaded_data['id'] = loaded_data['id'][0]
    if len(loaded_data['notes'])==0:
        loaded_data['notes']=""
    else:
        loaded_data['notes']=loaded_data['notes'][0]

    for bodyloc in ['sternum', 'head', 'wrist', 'ankle']:
        bodyloc_data = dict() # data structure to feed cleaned data into
        sensors = loaded_data[bodyloc][0].dtype.names
        for sensor_name, sensor_data in zip(sensors, loaded_data[bodyloc][0][0]):
            bodyloc_data[sensor_name] = dict()
            field_names = sensor_data[0][0].dtype.names
            for sensor_field, field_data in zip(field_names, sensor_data[0][0]):
                bodyloc_data[sensor_name][sensor_field] = field_data[0]
                if sensor_field == 'fs':
                    bodyloc_data[sensor_name][sensor_field] = bodyloc_data[sensor_name][sensor_field][0]
        loaded_data[bodyloc] = bodyloc_data
    return loaded_data

def panPeakDetect(detection, fs):
    """
    Jiapu Pan and Willis J. Tompkins.
    A Real-Time QRS Detection Algorithm.
    In: IEEE Transactions on Biomedical Engineering
    BME-32.3 (1985), pp. 230–236.

    Original implementation by Luis Howell luisbhowell@gmail.com, Bernd Porr, bernd.porr@glasgow.ac.uk, DOI: 10.5281/zenodo.3353396
    """
    min_distance = int(0.25 * fs)

    signal_peaks = [0]
    noise_peaks = []

    SPKI = 0.0
    NPKI = 0.0

    threshold_I1 = 0.0
    threshold_I2 = 0.0

    RR_missed = 0
    indexes = []

    missed_peaks = []
    peaks = scipy.signal.find_peaks(detection,distance=min_distance)[0]

    thres_weight = 0.125

    for index, peak in enumerate(peaks):

        if peak>4*fs and threshold_I1>max(detection[peak-4*fs:peak]): # reset thresholds if we do not see any peaks anymore
            SPKI_n = max(detection[peak-4*fs:peak])
            NPKI = min(NPKI*SPKI_n/SPKI, np.percentile(detection[peak-4*fs:peak], 80))
            SPKI = SPKI_n
            threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
            threshold_I2 = 0.5 * threshold_I1



        if detection[peak] > threshold_I1 and (peak - signal_peaks[-1]) > 0.3 * fs:
            signal_peaks.append(peak)
            indexes.append(index)
            SPKI = thres_weight * detection[signal_peaks[-1]] + (1-thres_weight) * SPKI
            if RR_missed != 0:
                if signal_peaks[-1] - signal_peaks[-2] > RR_missed:
                    missed_section_peaks = peaks[indexes[-2] + 1:indexes[-1]]
                    missed_section_peaks2 = []
                    for missed_peak in missed_section_peaks:
                        if missed_peak - signal_peaks[-2] > min_distance and signal_peaks[
                            -1] - missed_peak > min_distance and detection[missed_peak] > threshold_I2:
                            missed_section_peaks2.append(missed_peak)

                    if len(missed_section_peaks2) > 0:
                        signal_missed = [detection[i] for i in missed_section_peaks2]
                        index_max = np.argmax(signal_missed)
                        missed_peak = missed_section_peaks2[index_max]
                        missed_peaks.append(missed_peak)
                        signal_peaks.append(signal_peaks[-1])
                        signal_peaks[-2] = missed_peak
            if len(signal_peaks)>100 and thres_weight>0.1:
                thres_weight = 0.0125


        else:
            noise_peaks.append(peak)
            NPKI = thres_weight * detection[noise_peaks[-1]] + (1-thres_weight) * NPKI

        threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
        threshold_I2 = 0.5 * threshold_I1

        if len(signal_peaks) > 8:
            RR = np.diff(signal_peaks[-9:])
            RR_ave = int(np.mean(RR))
            RR_missed = int(1.66 * RR_ave)

    signal_peaks.pop(0)

    return signal_peaks


def pan_tompkins_detector(unfiltered_ecg, sr):
    """
    Jiapu Pan and Willis J. Tompkins.
    A Real-Time QRS Detection Algorithm.
    In: IEEE Transactions on Biomedical Engineering
    BME-32.3 (1985), pp. 230–236.

    Original implementation by Luis Howell luisbhowell@gmail.com, Bernd Porr, bernd.porr@glasgow.ac.uk, DOI: 10.5281/zenodo.3353396
    """
    maxQRSduration = 0.150  # sec
    filtered_ecg = butter_bandpass_filter(unfiltered_ecg, 5, 15, sr, order=1)

    diff = np.diff(filtered_ecg)
    squared = diff * diff

    mwa = scipy.ndimage.uniform_filter1d(squared, size=int(maxQRSduration * sr))
    # cap mwa during motion artefacts to make sure it does not screw the thresholds
    maxvals = scipy.ndimage.maximum_filter1d(filtered_ecg, size=int(maxQRSduration * sr))[:-1]/400
    mwa = np.asarray([v if v < maxval else maxval for maxval, v in zip(maxvals, mwa)])

    mwa[:int(maxQRSduration * sr * 2)] = 0

    searchr = int(maxQRSduration * sr)
    peakfind = butter_bandpass_filter(unfiltered_ecg, 7.5, 20, sr, order=1)

    mwa_peaks = panPeakDetect(mwa, sr)
    r_peaks2 = []
    for rp in mwa_peaks:
        r_peaks2.append(rp - searchr + np.argmax(peakfind[rp - searchr:rp + searchr + 1]))
    r_peaks3 = []
    for rp in r_peaks2:
        r_peaks3.append(rp - 2 + np.argmax(unfiltered_ecg[rp - 2:rp + 3])) # adjust by at most 2 samples to hit raw data max
    return np.asarray(r_peaks3)


def quotient_filter(hbpeaks, outlier_over=5, sampling_rate=128, tol=0.8):
    '''
    Function that applies a quotient filter similar to what is described in
    "Piskorki, J., Guzik, P. (2005), Filtering Poincare plots"
    it preserves peaks that are part of a sequence of [outlier_over] peaks with
    a tolerance of [tol]'''
    good_hbeats = []
    good_rrs = []
    good_rrs_x = []
    for i, peak in enumerate(hbpeaks[:-(outlier_over-1)]):
        hb_intervals = [hbpeaks[j]-hbpeaks[j-1]  for j in range(i+1, i+outlier_over)]
        hr = 60/((sum(hb_intervals))/((outlier_over-1)*sampling_rate))
        if min(hb_intervals) > max(hb_intervals)*tol and hr > 35 and hr < 185: # -> good data

            for p in hbpeaks[i:i+outlier_over]:
                if len(good_hbeats) == 0 or p > good_hbeats[-1]:
                    good_hbeats.append(p)
                    if len(good_hbeats) > 1:
                        rr = good_hbeats[-1]-good_hbeats[-2]
                        if rr<min(hb_intervals)/tol and rr>max(hb_intervals)*tol:
                            good_rrs.append(rr)
                            good_rrs_x.append((good_hbeats[-1]+good_hbeats[-2])/2)
    return np.array(good_hbeats), np.array(good_rrs), np.array(good_rrs_x)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = scipy.signal.butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.sosfiltfilt(sos, data)
    return y

#%%
ecghrs = dict()
ppghrs = dict()
for p in Path('./data/').iterdir():
    part_data = load_wildppg_participant(p.absolute())
    ppghrs[part_data['id']] = dict()
    r_peaks = pan_tompkins_detector(part_data['sternum']['ecg']['v'], part_data['sternum']['ecg']['fs'])
    ecgpks_filt, rrs, rrxs = quotient_filter(r_peaks, outlier_over=5, tol=0.75)

    hrs = []
    for win_s in range(0, max(ecgpks_filt), winsize * part_data['sternum']['ecg']['fs']):
        rr_in_win = rrs[np.logical_and(rrxs > win_s, rrxs < win_s + winsize * part_data['sternum']['ecg']['fs'])]
        if len(rr_in_win) > 1:  # at least 2
            hrs.append(60 * len(rr_in_win) / (np.sum(rr_in_win) / part_data['sternum']['ecg']['fs']))
        else:
            hrs.append(np.nan)
    ecghrs[part_data['id']] = np.array(hrs)


    for body_loc, file_idx in zip(['sternum', 'head', 'ankle', 'wrist'], list(range(1, 5))):
        ppg_peaks_path = Path('./src/heuristic_baselines/processing_files/ppg-beats-processing-{}/000{}_ppg_beats.mat'.format(part_data['id'], file_idx))
        assert ppg_peaks_path.exists(), ("The .mat file is missing with the ppg peaks for participant {}, {}. "
                                         "First generate them using the matlab script. May also be a path issue."
                                         "File expected at path {}").format(part_data['id'], body_loc, ppg_peaks_path.absolute())
        ppg_peaks = scipy.io.loadmat(ppg_peaks_path)['ppg_beats_inds']

        ppghrs[part_data['id']][body_loc] = dict()
        for method, pks in zip(ppg_peaks[0].dtype.names, ppg_peaks[0][0]):
            ppghrs[part_data['id']][body_loc][method] = []
            ppgpks_filt, prrs, prrxs = quotient_filter(pks - 1, outlier_over=5, tol=0.6)  # change to python indexing, filter outliers
            for win_s in range(0, max(ecgpks_filt), winsize * part_data[body_loc]['ppg_g']['fs']): # check in 8-second-windows
                prrmsk = np.logical_and(prrxs > win_s, prrxs < win_s + winsize * part_data[body_loc]['ppg_g']['fs'])
                if np.sum(prrmsk) > 0: # at least 1
                    ppghrs[part_data['id']][body_loc][method].append(60 * len(prrs[prrmsk]) / (np.sum(prrs[prrmsk]) / part_data[body_loc]['ppg_g']['fs']))
                else:
                    ppghrs[part_data['id']][body_loc][method].append(np.nan)
            ppghrs[part_data['id']][body_loc][method] = np.array(ppghrs[part_data['id']][body_loc][method])



        for method in ppghrs[part_data['id']][body_loc]:
            outdict['part'].append(part_data['id'])
            outdict['body_loc'].append(body_loc)
            outdict['method'].append(method)

            hrsmsk = np.logical_and(~np.isnan(ppghrs[part_data['id']][body_loc][method]), ~np.isnan(ecghrs[part_data['id']]))
            outdict['coverage'].append(np.sum(hrsmsk)/np.sum(~np.isnan(ecghrs[part_data['id']])))

            # exclude recording interruptions
            if part_data['id']=="l38" and body_loc=="wrist":
                gap_mask = np.full(len(hrsmsk), True)
                (s, e) = (4337572, 4545543)
                gap_mask[s//(part_data[body_loc]['ppg_g']['fs']*winsize):1+e//(part_data[body_loc]['ppg_g']['fs']*winsize)] = False
                hrsmsk = np.logical_and(hrsmsk, gap_mask)
            elif part_data['id']=="k2s" and body_loc=="wrist":
                gap_mask = np.full(len(hrsmsk), True)
                (s, e) = (3786939, 4194810)
                gap_mask[s//(part_data[body_loc]['ppg_g']['fs']*winsize):1+e//(part_data[body_loc]['ppg_g']['fs']*winsize)] = False
                hrsmsk = np.logical_and(hrsmsk, gap_mask)
            elif part_data['id']=="an0" and body_loc=="ankle":
                gap_mask = np.full(len(hrsmsk), True)
                (s, e) = (284543, 286400)
                gap_mask[s//(part_data[body_loc]['ppg_g']['fs']*winsize):1+e//(part_data[body_loc]['ppg_g']['fs']*winsize)] = False
                hrsmsk = np.logical_and(hrsmsk, gap_mask)

            pred = ppghrs[part_data['id']][body_loc][method][hrsmsk]
            targ = ecghrs[part_data['id']][hrsmsk]
            outdict['MAE'].append(np.mean(np.abs(pred-targ)))
            outdict['MAPE'].append(np.mean(np.abs(pred-targ)/targ))
            outdict['RMSE'].append(np.sqrt(((pred - targ) ** 2).mean()))
            outdict['corr'].append(np.corrcoef(pred, targ)[0][1])
            outdict['dataset'].append("WildPPG")


#%%
rslts = pd.DataFrame.from_dict(outdict)
rslts.to_csv("heuristic_results_final.csv", index=False)

#%%
print(rslts[rslts["body_loc"]=="wrist"].groupby(["method"]).mean()[['MAE', 'RMSE', 'corr']])
