import os
import librosa
import soundfile as sf
import numpy as np


# https://medium.com/@keur.plkar/audio-data-augmentation-in-python-a91600613e47
# I used the above link when determining which types of data augmentation to implement.
def augmentate(dataset_basedir, augment_type):
    """
    augmentate() will take a dataset base directory, and generate data from the original training set dependent
    on the type of augmentation desired.
    """
    # specify type of file to be augmented
    ext = ".wav"
    # set up training directory
    train_dir = dataset_basedir + "train/"
    # set up augmentation directory
    augment_type_dir = train_dir + "augmented_data/" + augment_type + "/"
    # ensure that directory for augmented data exists.  If not, create the directory
    if not os.path.isdir(train_dir + "augmented_data/"):
        os.mkdir(train_dir + "augmented_data/")
    # if the augmented data has already been generated, exit
    if os.path.isdir(augment_type_dir):
        return
    # create directory where augmented data will be stored
    os.mkdir(augment_type_dir)
    # file_id will be used to create unique file names
    file_id = 0
    # loop through original training set
    for root, dirs, files in os.walk(train_dir):
        # do not augment data that is already augmented
        if "augmented_data" in root:
            continue
        # for each file in original training set
        for f in files:
            _, file_ext = os.path.splitext(f)
            file_ext.lower()
            # if file is a wav file
            if file_ext == ext:
                # load the wav file
                wav, sample_rate = librosa.load(os.path.join(root, f), sr=None)
                # do noise addition
                if augment_type == "add_noise":
                    wav = add_noise(wav)
                # else do time stretching
                elif augment_type == "stretch_time":
                    wav = stretch_time(wav)
                # or else do pitch shifting
                elif augment_type == "shift_pitch":
                    wav = shift_pitch(wav, sample_rate)
                # output the new wav file generated
                new_f = f[0:2] + str(file_id) + ext
                sf.write(os.path.join(augment_type_dir, new_f), wav, sample_rate)
                file_id += 1
    return


# this function will add noise to wav data
def add_noise(wav):
    wav = np.add(wav, 0.0001 * np.random.normal(0, 1, len(wav)))
    return wav


# this function will time stretch wav data
def stretch_time(wav):
    stretch_factor = np.random.uniform(low=0.8, high=1, size=1)
    wav = librosa.effects.time_stretch(wav, stretch_factor[0])
    return wav


# this function will shift the pitch of wav data.  The desired sample rate is needed.
def shift_pitch(wav, sample_rate):
    pitch_factor = np.random.uniform(low=-3.0, high=3, size=1)
    wav = librosa.effects.pitch_shift(wav, sample_rate, n_steps=pitch_factor[0])
    return wav
