import os
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split


def model_constructor(inp_shape, n_speakers: int, learning_rate: float):
    """
    BUILD THE MODEL ARCHITECTURE.
    :param inp_shape: input dimensions
    :param n_speakers: number of speaker classes
    :param learning_rate: init learning rate parameter
    :return model: compiled model
    """

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=inp_shape, name='input'),           # input layer
        tf.keras.layers.Dense(256, activation='relu', name='hidden1'),          # 1st dense layer
        tf.keras.layers.Dense(64, activation='relu', name='hidden2'),           # 2nd dense layer
        tf.keras.layers.Dense(n_speakers, activation='softmax', name='output')  # output layer
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    model.summary()
    return model


def data_format(data, mode: str):
    """
    CHANNELS FIRST (NCHW) OR CHANNELS LAST (NHWC) INPUT REPRESENTATION.
    :param data: input data
    :param mode: CPU or GPU
    :return data: input data in NCHW or NHWC format
    """

    if mode == 'gpu':
        data = np.expand_dims(data, 1)  # channels first
    else:  # 'cpu'
        data = np.expand_dims(data, 3)  # channels last
    return data


def data_shaper(data_set, target_set, mode: str):
    """
    SHAPE THE MODEL INPUTS.
    :param data_set: OpenL3 embeddings corresponding to the audio inputs
    :param target_set: labels corresponding to data_set
    :param mode: CPU or GPU for data_format
    :return data_total: input data shaped by frames
    :return targets_total: target labels shaped by frames
    """

    n_input_frames = 1
    n_features = data_set[0].shape[1]
    data_total = []
    targets_total = []
    for n_file in range(len(data_set)):
        time_steps = data_set[n_file].shape[0] - n_input_frames
        file_segments = np.zeros((n_input_frames, n_features, time_steps)).astype(np.float32)
        for step in range(time_steps):
            segment = data_set[n_file][step:(step + n_input_frames), :]
            file_segments[:, :, step] = segment
        file_segments = np.transpose(file_segments, (2, 0, 1))
        # file_segments = data_format(file_segments, mode)
        file_targets = np.ones(time_steps).astype(np.float32)
        file_targets *= target_set[n_file]
        data_total.extend(file_segments)
        targets_total.extend(file_targets)
    data_total = np.array(data_total)
    targets_total = np.array(targets_total)
    return data_total, targets_total


def data_loader(path: str):
    """
    READ THE DATA FROM PATH.
    :param path: path to the data (OpenL3 embeddings)
    :return data_total: list of data predictors or label targets
    :return labels_total: list of target speaker label which is several first numbers in filename before "-"
    """

    data_total = []  # list of data samples
    labels_total = []  # list of target speaker labels
    n_file = 0  # index of file
    label_int = 0  # init label value
    files_list = os.listdir(path)  # list of input files
    for filename in files_list:
        data = np.load(os.path.join(path, filename), allow_pickle=True)
        label_str = '-'.join(filename.split('-')[:-2])  # such labels make sense only for LibriSpeech filenames
        if filename.endswith('.npz'):  # predictors
            emb, ts = data['embedding'], data['timestamps']
            data_total.append(emb)
            if n_file == 0:
                labels_total.append(label_int)
            else:
                label_str_prev = '-'.join(files_list[n_file-1].split('-')[:-2])
                if label_str_prev != label_str:
                    label_int += 1
                labels_total.append(label_int)
            n_file += 1
        else:
            # data_total.append(data)  # targets
            raise Exception('Only .npz must be processed.')

    return data_total, labels_total


def main(path_data: str, path_model: str, mode: str, n_epochs: int, n_batch: int):
    """
    :param path_data: directory of data inputs
    :param path_model: path to save trained model
    :param mode: CPU or GPU computations
    :param n_epochs: maximum number of training epochs
    :param n_batch: mini batch size
    :return: trained model is saved to path_model
    """

    # LOAD DATA
    print('Load dataset...')
    train_inputs, train_targets = data_loader((os.path.join(path_data, 'train', 'predictors')))
    # [train_targets] = data_loader((os.path.join(path_data, 'train', 'targets')))
    # PREPARE INPUTS
    X, y = data_shaper(
        data_set=train_inputs,
        target_set=train_targets,
        mode=mode,
    )
    # Split training and validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
    # DEFINE MODEL
    print('Define model structure...')
    model = model_constructor(
        inp_shape=(X.shape[1], X.shape[2]),
        n_speakers=len(np.unique(train_targets)),
        learning_rate=0.0001,
    )
    # TRAIN MODEL
    print('Start training...')
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=n_epochs,
        batch_size=n_batch,
    )
    # SAVE MODEL
    model.save(path_model, 'model.model')
    print('Model saved.')


if __name__ == '__main__':
    """
    GET VOICE-PRINT CLASSIFIER.
    """

    # Set random seeds
    os.environ['PYTHONHASHSEED'] = str(1)
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)
    # GPU settings
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("TensorFlow IS using the GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)  # memory management for GPU
        computation_mode = 'gpu'
    else:
        print("TensorFlow IS NOT using the GPU")
        computation_mode = 'cpu'
    # Define params
    path_root = os.getcwd()  # project directory
    path_in = os.path.join(path_root, 'embeddings')  # specify data directory
    path_out = os.path.join(path_root, 'model')  # specify output model directory
    if not os.path.isdir(path_out):
        os.makedirs(path_out)  # make output model directory
    # Run training
    print("\nRun VP based on OpenL3 training process\n")
    main(
        path_data=path_in,
        path_model=path_out,
        mode=computation_mode,
        n_epochs=23,  # number of training epochs
        n_batch=32,   # mini batch size
    )
    print("\nDone!")
