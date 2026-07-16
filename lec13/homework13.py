import numpy as np
import librosa


def lpc(speech, frame_length, frame_skip, order):
    """
    Perform linear predictive analysis of input speech.
    """

    nframes = int((len(speech) - frame_length) / frame_skip)

    A = np.zeros((nframes, order + 1))
    excitation = np.zeros((nframes, frame_length))

    for i in range(nframes):
        start = i * frame_skip
        frame = speech[start:start + frame_length]

        # LPC coefficients
        a = librosa.lpc(frame, order=order)
        A[i] = a

        residual = np.copy(frame)

        for n in range(order, frame_length):
            prediction = 0.0
            for k in range(1, order + 1):
                prediction -= a[k] * frame[n - k]
            residual[n] = frame[n] - prediction

        excitation[i] = residual

    return A, excitation


def synthesize(e, A, frame_skip):
    """
    Synthesize speech from LPC residual and coefficients.
    """

    order = A.shape[1] - 1
    nframes = len(A)

    synthesis = np.zeros(nframes * frame_skip)

    for i in range(nframes):
        frame = e[i * frame_skip:(i + 1) * frame_skip].copy()
        a = A[i]

        for n in range(frame_skip):
            for k in range(1, min(order, n) + 1):
                frame[n] -= a[k] * frame[n - k]

        synthesis[i * frame_skip:(i + 1) * frame_skip] = frame

    return synthesis


def robot_voice(excitation, T0, frame_skip):
    """
    Create robot voice excitation.
    """

    nframes = excitation.shape[0]

    gain = np.sqrt(np.mean(excitation ** 2, axis=1))

    e_robot = np.zeros(nframes * frame_skip)

    for i in range(nframes):
        frame = np.zeros(frame_skip)
        frame[::T0] = gain[i]
        e_robot[i * frame_skip:(i + 1) * frame_skip] = frame

    return gain, e_robot




speech, fs = librosa.load("lec13_speech_waveform.wav", sr=8000)

frame_length = int(fs * 0.025)     # 200
frame_skip = int(fs * 0.01)        # 80
order = 10

# LPC Analysis
A, excitation = lpc(speech, frame_length, frame_skip, order)

# Create excitation signal
e = np.hstack(excitation[:, frame_length-frame_skip:])

# Synthesize speech
synthesis = synthesize(e, A, frame_skip)

# Robot Voice
T0 = int(fs / 100)
gain, e_robot = robot_voice(excitation, T0, frame_skip)

#output

print("="*60)
print("LPC ANALYSIS")
print("="*60)
print("Speech Length:", len(speech))
print("Sampling Rate:", fs)
print("Number of Frames:", len(A))
print("Frame Length:", frame_length)
print("Frame Skip:", frame_skip)
print("LPC Order:", order)

print("\nA Shape:", A.shape)
print("Excitation Shape:", excitation.shape)

print("\nFirst LPC Coefficient Vector:")
print(A[0])

print("\nFirst 20 Excitation Samples:")
print(excitation[0][:20])

print("\n" + "="*60)
print("SYNTHESIS")
print("="*60)

print("Excitation Signal Length:", len(e))
print("Synthesized Speech Length:", len(synthesis))
print("Synthesized Power:",
      np.mean(synthesis**2))

print("\nFirst 20 Synthesized Samples:")
print(synthesis[:20])

print("\n" + "="*60)
print("ROBOT VOICE")
print("="*60)

print("Gain Shape:", gain.shape)
print("Robot Excitation Length:", len(e_robot))

print("\nFirst 10 Gain Values:")
print(gain[:10])

print("\nFirst 80 Robot Excitation Samples:")
print(e_robot[:80])

print("\nFinished Successfully.")
