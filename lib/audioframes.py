import numpy as np
import soundfile as sf


# Code by Joel Bradley
class AudioFrames:
    """AudioFrames
    A class for iterating over frames of audio data
    """
    
    def __init__(self, filename, adv_ms, len_ms):
        """"AudioFrames(filename, adv_ms, len_ms)
        Create a stream of audio frames where each is in len_ms milliseconds long
        and frames are advanced by adv_ms.              
        """
        # read in wave file
        self.data, self.Fs = sf.read(filename)
        # set the frame length in ms
        self.len_ms = len_ms
        # set the frame length in samples
        self.len_N = int((self.Fs * self.len_ms) / 1000)
        # set the frame advance in ms
        self.adv_ms = adv_ms
        # set the frame advance in samples
        self.adv_N = int((self.Fs * self.adv_ms) / 1000)
        # set current sample iteration
        self.current_sample = 0

    def get_framelen_samples(self):
        "get_framelen_ms - Return frame length in samples"
        return self.len_N
    
    def get_framelen_ms(self):
        "get_framelen_ms - Return frame length in ms"
        return self.len_ms
    
    def get_frameadv_samples(self):
        "get_frameadv_ms - Return frame advance in samples"
        return self.adv_N  

    def get_frameadv_ms(self):
        "get_frameadv_ms - Return frame advance in ms"
        return self.adv_ms
    
    def get_Fs(self):
        "get_Fs() - Return sample rate"
        return self.Fs
    
    def get_Nyquist(self):
        raise NotImplemented
    
        
    def __iter__(self):
        """"iter() - Return a frame iterator
        WARNING:  Multiple iterators on same soundfile are not guaranteed to work as expected"""
        # return instance of self as iterator
        return self

    def __next__(self):
        """"next() - Return next frame"""
        # make sure that next frame does not exceed wav file limits
        if self.current_sample + self.len_N > np.size(self.data, 0):
            # if so, reset current sample iteration
            self.current_sample = 0
            raise StopIteration
        # grab next frame by starting at current sample
        current_frame = self.data[self.current_sample:self.current_sample+self.len_N]
        # advance the sample iteration
        self.current_sample += self.adv_N
        return current_frame

