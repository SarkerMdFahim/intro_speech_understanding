import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord.
    '''
    duration = 0.5
    t = np.arange(0, duration, 1/Fs)

    # Root tone
    root = np.sin(2 * np.pi * f * t)

    # Major third (4 semitones above)
    major_third_freq = f * (2 ** (4/12))
    major_third = np.sin(2 * np.pi * major_third_freq * t)

    # Perfect fifth (7 semitones above)
    perfect_fifth_freq = f * (2 ** (7/12))
    perfect_fifth = np.sin(2 * np.pi * perfect_fifth_freq * t)

    # Combine tones
    x = root + major_third + perfect_fifth

    return x


def dft_matrix(N):
    '''
    Create an NxN DFT matrix.
    '''
    n = np.arange(N)
    k = n.reshape((N, 1))

    W = np.exp(-2j * np.pi * k * n / N)

    return W


def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.
    '''
    N = len(x)

    # FFT
    X = np.fft.fft(x)

    # Magnitude spectrum
    magnitude = np.abs(X)

    # Frequency axis
    freqs = np.fft.fftfreq(N, d=1/Fs)

    # Keep only positive frequencies
    positive = freqs > 0
    freqs = freqs[positive]
    magnitude = magnitude[positive]

    # Find 3 largest peaks
    peak_indices = np.argsort(magnitude)[-3:]

    loudest_freqs = np.sort(freqs[peak_indices])

    f1, f2, f3 = loudest_freqs

    return f1, f2, f3


# Example Test
Fs = 8000
f = 440

x = major_chord(f, Fs)

f1, f2, f3 = spectral_analysis(x, Fs)

print("Three loudest frequencies:")
print(f1, f2, f3)
