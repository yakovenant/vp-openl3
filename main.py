import os
import librosa
import soundfile as sf
import openl3
import numpy as np
import warnings


def main(dir_inputs, sr_target):
    """
    Voiseprint based on OpenL3 test.
    :param dir_inputs: ...
    :param sr_target: ...
    :return: ...
    """

    print("\nRun VP based on OpenL3 test\n")
    list_audio = []  # list of input audio samples
    for name_audio in os.listdir(dir_inputs):
        print(f"Read {name_audio}")
        """Load mono audio inputs"""
        dir_audio = os.path.join(dir_inputs, name_audio)
        audio, sr = librosa.load(dir_audio, sr=None, mono=True)
        # audio, sr = sf.read(dir_audio)
        """Check sampling rate"""
        if sr > sr_target:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sr_target)
        elif sr < sr_target:
            warnings.warn("Skip audio input with low sampling rate.")
            continue
        list_audio.append(audio)
    """Compute embeddings"""
    print("\nCompute embeddings...")
    list_emb, list_tr = openl3.get_audio_embedding(list_audio, sr_target,
                                                   batch_size=32,
                                                   content_type="env",  # env, music
                                                   input_repr="mel128",  # linear, mel128, mel256
                                                   embedding_size=512  # 512, 6144
                                                   )
    print('done')


if __name__ == '__main__':

    """Define input params"""
    # path_root = os.getcwd()
    path_in = "D:\\Denoiserdata\\audio_test\\00"  # specify audio input directory
    sr = 16000  # target sampling rate for audio input

    main(dir_inputs=path_in, sr_target=sr)
    print("Done")
