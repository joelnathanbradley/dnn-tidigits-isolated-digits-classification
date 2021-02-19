from lib.audioframes import AudioFrames
from lib.dftstream import DFTStream
from lib.endpointer import Endpointer

from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np


# I used the following as a guide to build the Generator class
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class Generator(Sequence):
    """
    A batch generator class to incrementally load in data for neural network training
    """

    def __init__(self, X, y, adv_ms, len_ms, avg_frames_per_speech_duration, batch_size, shuffle=True):
        """
        Store the data audio names, their labels, the frame length and advance in ms, the desired amount of audio frames
        in each audio file to use for training, the batch size, and whether to shuffle the data after each epoch or not.
        """

        self.adv_ms = adv_ms
        self.len_ms = len_ms
        self.avg_frames_per_speech_duration = avg_frames_per_speech_duration
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.X))
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        """
        Grab a batch of data, and their labels
        """

        # Determine which data to grab in batch by determining indices
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        # Make temp dft stream to determine the appropriate size for X
        dft_stream_tmp = DFTStream(AudioFrames(self.X[0], self.adv_ms, self.len_ms))
        # initialize X and y
        X = np.empty([self.batch_size, self.avg_frames_per_speech_duration * dft_stream_tmp.size()])
        y = np.empty([self.batch_size, self.y.shape[1]])
        # for each audio file in batch
        for index, data_sample in enumerate(indexes):
            # create a dft stream and prepare for speech detection
            dft_stream = DFTStream(AudioFrames(self.X[data_sample], self.adv_ms, self.len_ms))
            dft = np.ones(dft_stream.shape()[0])
            for d in dft_stream:
                dft = np.vstack((dft, d))
            dft = dft[1:]
            # predict where speech occurs in audio file
            end_pointer = Endpointer(dft)
            predictions = end_pointer.predict(dft)
            # we ony want to grab speech, so determine which audio frame the speech begins
            first_frame = np.where(predictions)[0][0]
            # we have to have a fixed amount of frames, so make sure that we do not go out of bounds
            # this would occur if speech begins late in the audio file
            if first_frame + self.avg_frames_per_speech_duration > dft.shape[0]:
                first_frame = dft.shape[0] - self.avg_frames_per_speech_duration
            # grab the fixed amount of audio frames
            dft = dft[first_frame:first_frame + self.avg_frames_per_speech_duration, :]
            # loss blows up with negative values, so make power spectrum values positive
            dft = np.abs(dft)
            # normalize the data for better fitting
            dft = dft / dft.max(axis=0)
            # flatten the dft stream into a single row to add to X
            dft = dft.flatten()
            # insert the features and label
            X[index, :] = dft
            y[index, :] = self.y[data_sample]
        return X, y

    def __len__(self):
        """
        Determine the number of batches per epoch
        """

        return len(self.X) // self.batch_size

    def on_epoch_end(self):
        """
        At the end of each epoch, shuffle the data indices for better fitting
        """

        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

