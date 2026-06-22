import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------
# Feature Extraction
# --------------------------------------------------
def get_features(waveform, Fs):
    '''
    Get features and labels from waveform.
    '''

    # -------------------------
    # Pre-emphasis
    # -------------------------
    preemph = 0.97
    emphasized = np.append(
        waveform[0],
        waveform[1:] - preemph * waveform[:-1]
    )

    # -------------------------
    # Spectrogram Features
    # 4ms frame, 2ms step
    # -------------------------
    frame_len = int(0.004 * Fs)
    frame_step = int(0.002 * Fs)

    frames = []
    for start in range(
        0,
        len(emphasized) - frame_len + 1,
        frame_step
    ):
        frame = emphasized[start:start+frame_len]

        spectrum = np.abs(np.fft.rfft(frame))

        frames.append(spectrum)

    features = np.array(frames)

    # Keep only low-frequency half
    features = features[:, :features.shape[1]//2]

    # -------------------------
    # VAD Labels
    # 25ms frame, 10ms skip
    # -------------------------
    vad_len = int(0.025 * Fs)
    vad_step = int(0.010 * Fs)

    energies = []

    for start in range(
        0,
        len(waveform) - vad_len + 1,
        vad_step
    ):
        frame = waveform[start:start+vad_len]
        energies.append(np.sum(frame**2))

    energies = np.array(energies)

    threshold = 0.1 * np.max(energies)

    speech_frames = energies > threshold

    labels_vad = np.zeros(len(energies), dtype=int)

    label_id = 1
    in_segment = False

    for i in range(len(speech_frames)):

        if speech_frames[i]:
            if not in_segment:
                in_segment = True
                current_label = label_id
                label_id += 1

            labels_vad[i] = current_label

        else:
            in_segment = False

    # Repeat each VAD label 5 times
    labels = np.repeat(labels_vad, 5)

    # Match feature length
    nframes = min(len(features), len(labels))

    features = features[:nframes]
    labels = labels[:nframes]

    return features, labels


# --------------------------------------------------
# Neural Network Training
# --------------------------------------------------
def train_neuralnet(features, labels, iterations):
    '''
    Train neural network.
    '''

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    nfeats = features.shape[1]
    nlabels = int(np.max(labels)) + 1

    model = nn.Sequential(
        nn.LayerNorm(nfeats),
        nn.Linear(nfeats, nlabels)
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01
    )

    lossvalues = []

    for i in range(iterations):

        optimizer.zero_grad()

        outputs = model(X)

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        lossvalues.append(loss.item())

    return model, np.array(lossvalues)


# --------------------------------------------------
# Neural Network Testing
# --------------------------------------------------
def test_neuralnet(model, features):
    '''
    Return softmax probabilities.
    '''

    X = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X)
        probabilities = torch.softmax(
            outputs,
            dim=1
        ).numpy()

    return probabilities


# ==================================================
# Example
# ==================================================
Fs = 16000

# Generate 2-second random waveform
duration = 2
waveform = np.random.randn(Fs * duration)

features, labels = get_features(waveform, Fs)

print("Features shape:", features.shape)
print("Labels shape:", labels.shape)

model, lossvalues = train_neuralnet(
    features,
    labels,
    iterations=100
)

print("\nFinal Loss:", lossvalues[-1])

probabilities = test_neuralnet(
    model,
    features
)

print("\nProbability shape:",
      probabilities.shape)

print("\nFirst 5 probability vectors:")
print(probabilities[:5])
