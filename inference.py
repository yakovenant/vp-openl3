import os
import random
import numpy as np
from numpy.linalg import norm
from tensorflow import keras
from train import data_loader


def main(path_inputs: str, path_model: str):
    """
    VOICE-PRINT INFERENCE.
    :param path_inputs: path to the directory of OpenL3 embeddings
    :param path_model:  path to the directory of classification model
    :return: decision about speakers on two inputs -- same or not
    """

    # LOAD MODEL
    model = keras.models.load_model(path_model)  # classifier model
    # LOOP THROUGH INPUTS
    vps = []  # voice-prints list
    for speaker in range(2):
        # LOAD RANDOM TEST INPUT OpenL3 EMBEDDINGS
        data_inp = np.array(data_loader(random.choice(os.listdir(path_inputs))))
        # TRUNCATE INPUT
        if speaker > 0:
            if len(data_inp) > len(vps[0]):
                ...
            elif len(data_inp) < len(vps[0]):
                ...
        vps.append(data_inp)

        # EXTRACT VOICE-PRINT REPRESENTATIONS FROM EMBEDDINGS USING THE MODEL
        model(vps[speaker])
        speaker_vp = keras.models.Model(
            inputs=model.inputs,
            outputs=model.get_layer(name='hidden2').output,
        )
    # ESTIMATE COSINE SIMILARITY
    cos_sim = np.dot(vps[0], vps[1]) / (norm(vps[0]) * norm(vps[1]))

    # EVALUATE EER
    ...


if __name__ == '__main__':
    """
    RUN INFERENCE TO EXTRACT MLP-BASED VOICE-PRINT REPRESENTATIONS OF EMBEDDINGS 
    AND EVALUATE EER USING COSINE SIMILARITY
    """

    # Define input params
    dir_root = os.getcwd()  # project directory
    dir_in = os.path.join(dir_root, 'embeddings', 'test', 'predictors')  # specify test data directory
    dir_clf = os.path.join(dir_root, 'embeddings', 'model')  # specify classifier directory
    # Run inference
    main(
        path_inputs=dir_in,
        path_model=dir_clf,
    )
    print('Done!')
