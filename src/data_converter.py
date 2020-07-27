# -*- coding:utf-8 -*-
import os
import glob
import mne
import numpy as np


event_id = {'Sleep stage W': 0,
            'Sleep stage 1': 1,
            'Sleep stage 2': 2,
            'Sleep stage 3/4': 3,
            'Sleep stage R': 4}


def sleep_physionet_converter(src_path, trg_path, duration=30):
    # Physionet Sleep Dataset Converter (Sleep-EDF expanded-1.0.0)
    # We used EEG Fpz-Cz channels
    # * Input  : Physionet Sleep Dataset (.edf)
    # * Output : Converted Dataset (.npy)

    psg_fnames = glob.glob(os.path.join(src_path, '*PSG.edf'))
    ann_fnames = glob.glob(os.path.join(src_path, '*Hypnogram.edf'))
    psg_fnames.sort()
    ann_fnames.sort()

    annotation_desc_2_event_id = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
                                  'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4}

    for psg_fname, ann_fname in zip(psg_fnames, ann_fnames):
        total_x, total_y = [], []

        raw = mne.io.read_raw_edf(psg_fname, preload=True)
        ann = mne.read_annotations(ann_fname)
        raw.set_annotations(ann, emit_warning=True)
        raw.set_channel_types(mapping={'EOG horizontal': 'eog',
                                       'Resp oro-nasal': 'misc',
                                       'EMG submental' : 'misc',
                                       'Temp rectal'   : 'misc',
                                       'Event marker'  : 'misc'})

        raw.pick_channels(ch_names=['EEG Fpz-Cz', 'EOG horizontal', 'EMG submental'])
        event, _ = mne.events_from_annotations(
            raw=raw, event_id=annotation_desc_2_event_id, chunk_duration=duration
        )

        t_max = 30. - 1. / raw.info['sfreq']  # t_max in included
        try:
            epochs = mne.Epochs(raw=raw, events=event,
                                event_id=event_id, tmin=0., tmax=t_max, baseline=None)
        except ValueError:
            continue

        for epoch, event in zip(epochs, epochs.events):
            total_x.append(epoch)
            total_y.append(event[-1])

        total_x = np.array(total_x)
        total_y = np.array(total_y)

        # Saving Numpy Array
        name = os.path.basename(psg_fname).split('-')[0].lower()
        np_path = os.path.join(trg_path, name)
        np.savez(np_path, x=total_x, y=total_y)


if __name__ == '__main__':
    sleep_physionet_converter(
        src_path=os.path.join('..', 'data', 'physionet-sleep-data'),
        trg_path=os.path.join('..', 'data', 'physionet-sleep-npy'),
        duration=30
    )
