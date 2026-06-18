import numpy as np

# --------------------------------------------------
# Helper Function
# --------------------------------------------------
def waveform_to_frames(waveform, frame_length, step):
    num_frames = 1 + (len(waveform) - frame_length) // step

    frames = np.zeros((num_frames, frame_length))

    for i in range(num_frames):
        start = i * step
        frames[i] = waveform[start:start + frame_length]

    return frames


# --------------------------------------------------
# Voice Activity Detection (VAD)
# --------------------------------------------------
def VAD(waveform, Fs):

    frame_length = int(0.025 * Fs)  # 25 ms
    step = int(0.010 * Fs)          # 10 ms

    frames = waveform_to_frames(waveform, frame_length, step)

    energy = np.sum(frames ** 2, axis=1)

    threshold = 0.10 * np.max(energy)

    speech_frames = energy > threshold

    segments = []
    start = None

    for i, active in enumerate(speech_frames):

        if active and start is None:
            start = i

        elif not active and start is not None:

            end = i

            seg_start = start * step
            seg_end = (end - 1) * step + frame_length

            segments.append(waveform[seg_start:seg_end])

            start = None

    if start is not None:
        seg_start = start * step
        segments.append(waveform[seg_start:])

    return segments


# --------------------------------------------------
# Convert Segments to Models
# --------------------------------------------------
def segments_to_models(segments, Fs):

    models = []

    frame_length = int(0.004 * Fs)   # 4 ms
    step = int(0.002 * Fs)           # 2 ms

    for segment in segments:

        emphasized = np.append(
            segment[0],
            segment[1:] - 0.97 * segment[:-1]
        )

        if len(emphasized) < frame_length:
            continue

        frames = waveform_to_frames(
            emphasized,
            frame_length,
            step
        )

        frames *= np.hamming(frame_length)

        spectrum = np.abs(
            np.fft.rfft(frames, axis=1)
        )

        log_spectrum = np.log(spectrum + 1e-10)

        half = log_spectrum.shape[1] // 2

        low_freq = log_spectrum[:, :half]

        model = np.mean(low_freq, axis=0)

        models.append(model)

    return models


# --------------------------------------------------
# Cosine Similarity
# --------------------------------------------------
def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a)
        * np.linalg.norm(b)
        + 1e-10
    )


# --------------------------------------------------
# Speech Recognition
# --------------------------------------------------
def recognize_speech(
        testspeech,
        Fs,
        models,
        labels):

    test_segments = VAD(testspeech, Fs)

    test_models = segments_to_models(
        test_segments,
        Fs
    )

    Y = len(models)
    K = len(test_models)

    sims = np.zeros((Y, K))

    test_outputs = []

    for k in range(K):

        best_score = -np.inf
        best_label = None

        for y in range(Y):

            sim = cosine_similarity(
                models[y],
                test_models[k]
            )

            sims[y, k] = sim

            if sim > best_score:
                best_score = sim
                best_label = labels[y]

        test_outputs.append(best_label)

    return sims, test_outputs


# ==================================================
# EXAMPLE
# ==================================================

Fs = 16000

t = np.linspace(0, 1, Fs, endpoint=False)

# Training word 1 ("HELLO")
train1 = np.sin(2 * np.pi * 440 * t)

# Training word 2 ("YES")
train2 = np.sin(2 * np.pi * 880 * t)

# Create models
segments1 = VAD(train1, Fs)
segments2 = VAD(train2, Fs)

model1 = segments_to_models(segments1, Fs)[0]
model2 = segments_to_models(segments2, Fs)[0]

models = [model1, model2]

labels = ["HELLO", "YES"]

# Test speech (similar to HELLO)
testspeech = np.sin(2 * np.pi * 440 * t)

# Recognition
sims, outputs = recognize_speech(
    testspeech,
    Fs,
    models,
    labels
)

print("Similarity Matrix:")
print(sims)

print("\nRecognized Labels:")
print(outputs)
