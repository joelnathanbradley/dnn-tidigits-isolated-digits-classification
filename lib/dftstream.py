import numpy as np
from scipy.signal import get_window

# Code by Joel Bradley
class DFTStream:
    '''
    DFTStream - Transform a frame stream to various forms of spectra
    '''


    def __init__(self, frame_stream, specfmt="dB"):
        '''
        DFTStream(frame_stream, specfmt)        
        Create a stream of discrete Fourier transform (DFT) frames using the
        specified sample frame stream. Only bins up to the Nyquist rate are
        returned except wehn specfmt == "complex":
        
        specfmt - DFT output:  
            "complex" - return complex DFT results
             "dB" [default] - return power spectrum 20log10(magnitude)
             "mag^2" - magnitude squared spectrum
        '''

        # Note, you can implement only the default specfmt if you choose
        self.frame_stream = frame_stream
        self.frame_iter = iter(frame_stream)
         
    def shape(self):
        "shape() - Return dimensions of tensor yielded by next()"
        return [self.get_Hz(), 1]

    def size(self):
        "size() - number of elements in tensor generated by iterator"
        return np.asarray(np.product(self.shape()))
    
    def get_Hz(self):
        "get_Hz(Nyquist) - Return frequency bin labels"
        # get number of samples in a frame.  This will equate the number of DFT frequencies
        frame_samples = self.frame_stream.get_framelen_samples()
        # if even, grab exactly half of the frequencies
        if frame_samples % 2 == 0:
            Nyquist_bin = frame_samples // 2
        # if odd, ensure that a whole number of frequencies selected
        else:
            Nyquist_bin = (frame_samples + 1) // 2
        return Nyquist_bin
            
    def __iter__(self):
        "__iter__() Return iterator for stream"
        # return self as iterator
        return self
    
    def __next__(self):
        "__next__() Return next DFT frame"

        # grab the next frame
        frame_data = next(self.frame_iter)
        # determine size of frame
        frame_size = frame_data.size
        # apply hamming window, needs scipy
        hamming_window = get_window("hamming", frame_size)
        # take fourier transform to obtain frequencies
        dft_frame = np.fft.fft(frame_data * hamming_window)
        # take absolute magnitude of frequencies, and then compute power spectrum
        mag_dB = 20*np.log10(np.abs(dft_frame))
        # using Nyquist frequency, get rid of duplicate information
        mag_to_Nyquist_dB = mag_dB[0:self.get_Hz()]
        return mag_to_Nyquist_dB