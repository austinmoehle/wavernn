"""
Run inference on a WAV file using a frozen WaveRNN model.
"""
import math
import os
import random
import time

import click
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

MEL_BANDS = 80
SAMPLE_RATE = 16000
SCALING = 0.185

# Frozen graph nodes.
OUTPUT_NODE = "Inference/Model/MuLawExpanding/mul_1:0"
INPUT_NODE = "IteratorGetNext:1"
TRAINING = "training:0"


@click.command()
@click.argument("wav")
@click.option("--model", default="models/frozen.pb", help="Frozen graph")
@click.option("--output", default="outputs/audio.wav", help="Output WAV audio")
def inference(wav, model, output):
    """
    Converts an input WAV file to an 80-band mel spectrogram, then runs
    inference on the spectrogram using a frozen graph.

    Writes the output to a WAV file.
    """
    data, sr = librosa.core.load(wav, sr=SAMPLE_RATE, mono=True)
    print("Length of audio: {:.2f}s".format(float(len(data))/sr))

    spectrogram = compute_spectrogram(data, sr)
    plot_spectrogram(spectrogram)

    audio = run_wavernn(model, spectrogram, output)
    librosa.output.write_wav(output, audio, sr=SAMPLE_RATE)
    print("Wrote WAV file:", os.path.abspath(output))


def compute_spectrogram(audio, sr):
    """
    Converts audio to an 80-band mel spectrogram.

    Args:
        audio: Raw audio data.
        sr:    Audio sample rate in Hz.

    Returns:
        80-band mel spectrogram, a numpy array of shape [frames, 80].
    """
    spectrogram = librosa.core.stft(audio, n_fft=2048, hop_length=400,
        win_length=1600)
    spectrogram = np.abs(spectrogram)
    spectrogram = np.dot(
        librosa.filters.mel(sr, 2048, n_mels=80, fmin=0, fmax=8000),
        spectrogram)
    spectrogram = np.log(spectrogram*SCALING + 1e-2)
    return np.transpose(spectrogram)


def run_wavernn(model, spectrogram, output):
    """
    Run inference using a frozen model.

    Args:
        model:       Frozen graph file, .pb format.
        spectrogram: 80-band mel spectrogram.
        output:      Output file.

    Returns:
        Output audio, 16 kHz sample rate.
    """
    # Pad the spectrograms (in the time dimension) before input.
    padding = 12
    spectrogram = np.pad(spectrogram, [[padding, padding], [0, 0]],
                         mode='constant')

    with tf.gfile.GFile(model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:
            tf.import_graph_def(graph_def, name="")
            print("Generating samples...")
            start_time = time.time()
            audio = session.run(OUTPUT_NODE, feed_dict={
                INPUT_NODE: [spectrogram],
                TRAINING: False,
            })
            elapsed = time.time() - start_time
            generated_seconds = audio.size / SAMPLE_RATE

    print("Generated {:.2f}s in {:.2f}s ({:.3f}x realtime)."
        .format(generated_seconds, elapsed, generated_seconds / elapsed))
    return audio


def plot_spectrogram(spectrogram):
    librosa.display.specshow(np.transpose(spectrogram), cmap="plasma")
    plt.tight_layout()
    plt.savefig("spectrogram.png", bbox_inches=None, pad_inches=0)
    plt.close()


if __name__ == '__main__':
    inference()
