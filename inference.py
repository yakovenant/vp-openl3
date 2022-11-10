import os
import random
import numpy as np
from sklearn import metrics
from numpy.linalg import norm
from tensorflow import keras
from collections import Counter


def main(path_inputs: str, path_model: str, threshold: float, n_tests: int):
    """
    VOICE-PRINT TEST: COMPARISON VOICES OF TWO SPEECH FILES
    :param path_inputs: path to the directory of OpenL3 embeddings
    :param path_model:  path to the directory of classification model
    :param threshold: threshold value for cosine similarity
    :param n_tests: number of pair-wise tests
    :return: decision about speakers on two inputs -- same or not
    """

    # LOAD MODEL
    model = keras.models.load_model(path_model)  # classifier model
    # LOOP THROUGH TESTS
    lbls_total = []  # target labels: 1 = same speakers, 0 = different speakers
    outs_total = []  # actual outputs
    for test in range(n_tests):
        print(f'Run test {test}')
        # LOAD INPUTS
        embs = []  # OpenL3 embeddings list
        speakers_list = []  # list of original speakers
        for speaker in range(2):
            # LOAD RANDOM PREPROCESSED INPUT OpenL3 EMBEDDINGS FROM THE TESTING SET
            filename = random.choice(os.listdir(path_inputs))
            speaker_id = '-'.join(filename.split('-')[:-2])
            print(f'Test file {speaker+1}: {"_".join(filename.split("_")[:-1])}')
            input_test_emb = np.load(os.path.join(path_inputs, filename), allow_pickle=True)
            input_test_emb = input_test_emb['embedding']
            # TRUNCATE TEST INPUT TO GET EQUAL LENGTHS
            if speaker > 0:
                if len(input_test_emb) < len(embs[0]):
                    # truncate first input embedding
                    embs[0] = np.delete(embs[0], slice(len(embs[0]) - len(input_test_emb)), 0)
                elif len(input_test_emb) > len(embs[0]):
                    # truncate second input embedding
                    input_test_emb = np.delete(input_test_emb, slice(len(input_test_emb) - len(embs[0])), 0)
            embs.append(input_test_emb)
            speakers_list.append(speaker_id)
        # ASSIGN TRUE POSITIVE LABELS FOR EACH FRAME
        if speakers_list[0] != speakers_list[1]:
            lbls = 0
        else:
            lbls = 1
        # LOOP THROUGH THE INPUTS
        vps = []  # voice-print representations
        for speaker in range(len(embs)):
            # EXTRACT VOICE-PRINT REPRESENTATIONS USING PENULT LAYER OF THE MODEL
            m_outs = [layer.output for layer in model.layers]
            m_acts = keras.models.Model(model.input, m_outs)
            acts = m_acts.predict(embs[speaker])
            speaker_vp = acts[-2]
            vps.append(speaker_vp)
        # ESTIMATE COSINE SIMILARITY
        outs = []
        for frame in range(len(vps[0])):
            vp_1 = vps[0][frame, :]
            vp_2 = vps[1][frame, :]
            cos_sim = np.dot(vp_1, vp_2) / (norm(vp_1) * norm(vp_2))
            if cos_sim >= threshold:
                output = 1  # same speakers
            else:
                output = 0  # different speakers
            outs.append(output)
        # DECISION
        result = Counter(outs)
        if result[0] > result[1]:
            outs_total.append(0)
        else:
            outs_total.append(1)
        lbls_total.append(lbls)
    # EVALUATE EER
    eer = 1 - metrics.roc_auc_score(lbls_total, outs_total)
    print(f'EER: {eer}')


if __name__ == '__main__':
    """
    RUN INFERENCE TO EXTRACT MLP-BASED VOICE-PRINT REPRESENTATIONS OF EMBEDDINGS 
    AND EVALUATE EER USING COSINE SIMILARITY
    """

    # Define input params
    dir_root = os.getcwd()  # project directory
    dir_in = os.path.join(dir_root, 'embeddings', 'test', 'predictors')  # specify test data directory
    dir_clf = os.path.join(dir_root, 'model')  # specify classifier directory
    # Run inference
    print('Run voice-print inference\n')
    main(
        path_inputs=dir_in,
        path_model=dir_clf,
        threshold=0.9,  # threshold value for cosine similarity
        n_tests=20,  # number of pair-wise tests
    )
    print('Done!')
