import os
import openl3
import librosa
import warnings
import numpy as np
import pickle
from openl3.models import load_audio_embedding_model
from soundfile import write
from sklearn.model_selection import train_test_split


def get_embeddings(dir_inp: [list, str], dir_out: str, flag: str, model):
    """
    COMPUTE AND SAVE EMBEDDINGS.
    :param dir_inp: str or list[str] -- audio inputs directory
    :param dir_out: str -- audio embeddings directory
    :param flag: str or None -- string to be appended to the output filename
    :param model: -- OpenL3 model
    """

    if dir_inp:
        print("\nCompute embeddings...")
        openl3.core.process_audio_file(
            filepath=dir_inp,
            output_dir=dir_out,
            suffix=flag,
            model=model,
            batch_size=32,
        )  # process all files in the arguments
    else:
        raise Exception('Audio input list is empty.')


def preprocess_audio(samples_inp, sr_current: int, sr_target: int, filename: str):
    """
    AUDIO INPUTS PREPROCESSING.
    :param samples_inp: ...
    :param sr_current: ...
    :param sr_target: ...
    :param filename: ...
    :return: samples_out: ...
    """

    # Check sampling rate
    if sr_current > sr_target:
        samples_inp = librosa.resample(samples_inp, orig_sr=sr, target_sr=sr_target)  # down-sampling
    elif sr_current < sr_target:
        warnings.warn(f'{filename} has too low sampling rate.')
    # Remove silence from clean speech
    clips = librosa.effects.split(samples_inp, top_db=25)
    samples_out = []
    for c in clips:
        data = samples_inp[c[0]:c[1]]
        samples_out.extend(data)
    samples_out = np.array(samples_out)
    # Normalize audio volume
    max_peak = np.max(np.abs(samples_out))
    ratio = 1 / max_peak
    samples_out *= ratio

    return samples_out


def get_audio_inputs(sr_target: int, dir_inp: str, list_speakers: [list, str]):
    """
    PREPARE AUDIO INPUTS FOR EMBEDDING EXTRACTION.
    :param sr_target: int -- target sampling rate
    :param dir_inp: str -- audio input directory
    :param list_speakers: list[str] -- list of speaker categories which are subdirectories of dir_inp
    :return:
    list_audio: list -- list of audio samples
    list_dir_audio: list -- list of directories for audio samples
    """

    list_audio = []  # list of audio samples
    list_dir_audio = []  # list of directories for audio files
    list_targets = []  # list of speakers corresponding to each audio file
    for speaker in list_speakers:
        dir_speaker = os.path.join(dir_inp, speaker)
        for root, dirs, files in os.walk(dir_speaker):
            for filename in files:
                if filename.endswith('.wav') or filename.endswith('.flac'):
                    dir_audio = os.path.join(root, filename)
                    # Load mono audio input
                    audio, sr = librosa.load(dir_audio, sr=None, mono=True)
                    # Preprocess audio
                    audio = preprocess_audio(
                        samples_inp=audio,
                        sr_current=sr,
                        sr_target=sr_target,
                        filename=filename
                    )
                    # Save preprocessed audio
                    if filename.endswith('.wav'):
                        write(dir_audio, data=audio, samplerate=sr_target, format='wav')
                    elif filename.endswith('.flac'):
                        write(dir_audio, data=audio, samplerate=sr_target, format='flac')
                    # Append paths
                    list_audio.append(audio)
                    list_dir_audio.append(dir_audio)
                    list_targets.append(speaker)
                else:
                    print(f'Skip {filename}')
    print('...done\n')
    return list_audio, list_dir_audio, list_targets


def save_pickle(path, name, params):
    ...

    name += '.pickle'
    with open(os.path.join(path, name), 'bw') as handle:
        pickle.dump(params, handle)


def check_dirs(path):
    """
    CREATE TRAINING DATA DIRECTORIES
    :param path: str -- path to extracted audio embeddings
    """
    if not os.path.isdir(path):
        os.makedirs(os.path.join(path, 'train', 'predictors'))
        os.makedirs(os.path.join(path, 'train', 'targets'))
        os.makedirs(os.path.join(path, 'test', 'predictors'))
        os.makedirs(os.path.join(path, 'test', 'targets'))


def main(dir_inputs: str, dir_outputs: str, sr_target: int, model):
    """
    :param dir_inputs: str -- ...
    :param dir_outputs: str -- ...
    :param sr_target: int -- ...
    :param model: -- ...
    """

    # DATA PREPARATION
    check_dirs(dir_outputs)  # prepare data directories
    # Gather all speaker categories
    list_speakers_total = []
    for _speaker in os.listdir(dir_inputs):
        list_speakers_total.append(_speaker)
    # Separate speaker categories into the training and testing subsets
    list_speakers_train, list_speakers_test = train_test_split(list_speakers_total, test_size=0.11)
    # Get training subset
    print('Get training audio subset...')
    _, list_dir_audio_train, list_targets_train = get_audio_inputs(
        sr_target=sr_target,
        dir_inp=dir_inputs,
        list_speakers=list_speakers_train
    )
    # Save train targets
    save_pickle(os.path.join(dir_outputs, 'train', 'targets'), 'targets_train', list_targets_train)
    # Get testing subset
    print('Get testing audio subset...\n')
    _, list_dir_audio_test, list_targets_test = get_audio_inputs(
        sr_target=sr_target,
        dir_inp=dir_inputs,
        list_speakers=list_speakers_test
    )
    # Save test targets
    save_pickle(os.path.join(dir_outputs, 'test', 'targets'), 'targets_test', list_targets_test)
    # COMPUTE AND SAVE EMBEDDINGS
    # For training subset
    print('\nCompute and save training embeddings\n')
    get_embeddings(
        dir_inp=list_dir_audio_train,
        dir_out=os.path.join(dir_outputs, 'train', 'predictors'),
        flag='train',
        model=model
    )
    # For testing subset
    print('\nCompute and save testing embeddings\n')
    get_embeddings(
        dir_inp=list_dir_audio_test,
        dir_out=os.path.join(dir_outputs, 'test', 'predictors'),
        flag='test',
        model=model
    )


if __name__ == '__main__':
    """
    EXTRACT AUDIO EMBEDDINGS FROM OpenL3 MODEL AND SAVE IT TO SPECIFIED DIRECTORY.
    """

    # Define params
    path_root = os.getcwd()  # project directory
    path_in = 'D:\\Data\\LibriSpeech\\train-clean-100'  # specify audio input directory
    path_out = os.path.join(path_root, 'embeddings')  # specify audio embeddings directory
    sr = 16000  # target sampling rate for audio input
    # Load model
    M = openl3.models.load_audio_embedding_model(
        input_repr='mel256',  # linear, mel128, mel256
        content_type='env',   # env, music
        embedding_size=512,   # 512, 6144
    )  # process inputs with specified params
    # Run embedding extractor
    print("\nRun VP based on OpenL3 audio embedding extractor\n")
    main(
        dir_inputs=path_in,
        dir_outputs=path_out,
        sr_target=sr,
        model=M
    )  # run
    print("\nDone!")
