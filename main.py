import os
import openl3
import librosa
# import soundfile as sf
import numpy as np
import warnings
from openl3.models import load_audio_embedding_model
from sklearn.model_selection import train_test_split


def check_params(path_in, path_out, sr):

    assert type(path_in) is str
    assert type(path_out) is str
    assert type(sr) is int


def main(dir_inputs, dir_outputs, sr_target, model):
    """
    Voiseprint based on OpenL3 test.
    :param dir_inputs:
    :param dir_outputs:
    :param sr_target:
    :param model:
    :return:
    """

    """TRAINING DATA GENERATION"""

    """Gather all speaker categories"""
    list_speakers = []
    for speaker in os.listdir(dir_inputs):
        list_speakers.append(speaker)
    """Separate speaker categories into the training and testing subsets"""
    list_speakers_train, list_speakers_test = train_test_split(list_speakers, test_size=0.11)
    """Process training subset"""
    ...
    list_audio = []  # list of input audio samples
    list_dir_audio = []  # list of directories for input audio samples
    for name_audio in os.listdir(dir_inputs):
        print(f"Read {name_audio}")
        """Load mono audio inputs"""
        dir_audio = os.path.join(dir_inputs, name_audio)
        audio, sr = librosa.load(dir_audio, sr=None, mono=True)
        # audio, sr = sf.read(dir_audio)
        """Check sampling rate"""
        if sr > sr_target:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sr_target)  # down-sampling
        elif sr < sr_target:
            warnings.warn("Skip audio input with low sampling rate.")
            continue
        list_audio.append(audio)
        list_dir_audio.append(dir_audio)
    """Compute and save embeddings"""
    print("\nCompute embeddings...")
    openl3.core.process_audio_file(
        filepath=list_audio,
        output_dir=dir_outputs,
        suffix=None,
        model=model,
        batch_size=32,
    )  # process all files in the arguments


if __name__ == '__main__':
    """
    ...
    """

    """Define input params"""
    path_root = os.getcwd()  # project directory
    path_in = 'D:\\Data\\LibriSpeech\\train-clean-100'  # specify audio input directory
    path_out = os.path.join(path_root, 'output')  # specify output directory
    if not os.path.isdir(path_out):  # create output directory
        os.makedirs(path_out)
    sr = 16000  # target sampling rate for audio input
    """Checking input params"""
    check_params(path_in, path_out, sr)
    """Load model"""
    model = openl3.models.load_audio_embedding_model(
        input_repr='mel128',  # linear, mel128, mel256
        content_type='env',   # env, music
        embedding_size=512,   # 512, 6144
        frontend='kapre'      # kapre, librosa
    )  # process inputs with specified params
    """Run inference"""
    print("\nRun VP based on OpenL3 test\n")
    main(dir_inputs=path_in, dir_outputs=path_out, sr_target=sr, model=model)
    print("Done")
