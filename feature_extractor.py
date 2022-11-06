import os
import openl3
import librosa
# import soundfile as sf
import numpy as np
import warnings
from openl3.models import load_audio_embedding_model
from sklearn.model_selection import train_test_split


def main(format_audio: str, dir_inputs: str, dir_outputs: str, sr_target: int, model):
    """
    Voiseprint based on OpenL3 test.
    :param format_audio: str
    :param dir_inputs: str
    :param dir_outputs: str
    :param sr_target: int
    :param model:
    :return:
    """

    """AUDIO INPUT PREPROCESSING"""
    ...
    """TRAINING DATA GENERATION"""
    """Gather all speaker categories"""
    list_speakers_total = []
    for speaker in os.listdir(dir_inputs):
        list_speakers_total.append(speaker)
    """Separate speaker categories into the training and testing subsets"""
    list_speakers_train, list_speakers_test = train_test_split(list_speakers_total, test_size=0.11)
    """Process training subset"""
    ...
    list_audio_train = []  # list of training audio samples
    list_dir_audio_train = []  # list of directories for training audio samples
    for speaker in list_speakers_train:
        _dir_speaker = os.path.join(dir_inputs, speaker)
        for root, dirs, files in os.walk(_dir_speaker):
            for _name_audio in files:
                if _name_audio.endswith(format_audio):
                    _dir_audio = os.path.join(root, _name_audio)
                    print(f'Read {_dir_audio}')
                    """Load mono audio input"""
                    _audio, _sr = librosa.load(_dir_audio, sr=None, mono=True)
                    # _audio, _sr = sf.read(_dir_audio)
                    """Check sampling rate"""
                    if _sr > sr_target:
                        _audio = librosa.resample(_audio, orig_sr=sr, target_sr=sr_target)  # down-sampling
                    elif _sr < sr_target:
                        warnings.warn("Skip audio input with low sampling rate.")
                        continue
                    list_audio_train.append(_audio)
                    list_dir_audio_train.append(_dir_audio)
                else:
                    print(f'Skip {_name_audio}')
    ...
    """Compute and save embeddings"""
    if list_dir_audio_train:
        print("\nCompute embeddings...")
        openl3.core.process_audio_file(
            filepath=list_dir_audio_train,
            output_dir=dir_outputs,
            suffix=None,
            model=model,
            batch_size=32,
        )  # process all files in the arguments
    else:
        raise Exception('Audio input list is empty.')


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
    """Load model"""
    M = openl3.models.load_audio_embedding_model(
        input_repr='mel128',  # linear, mel128, mel256
        content_type='env',   # env, music
        embedding_size=512,   # 512, 6144
        frontend='kapre'      # kapre, librosa
    )  # process inputs with specified params
    """Run inference"""
    print("\nRun VP based on OpenL3 test\n")
    main(
        format_audio='.wav',
        dir_inputs=path_in,
        dir_outputs=path_out,
        sr_target=sr,
        model=M
    )  # run
    print("Done!")
