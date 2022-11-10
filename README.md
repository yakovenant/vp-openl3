# vp-openl3
This is voice-print test project based on OpenL3 and TensorFlow.

1) Specify the path to the speech dataset and run embeddings.py to get embeddings from OpenL3 model.
2) Run train.py to train MLP-based speaker classifier using OpenL3 audio embeddings.
3) Run inference.py to extract MLP-based representations of embeddings and estimate EER using cosine similarity.

One can get test inputs by embeddings.py and run inference without training: "model" directory contains example of trained classifier.

TODO:
1) Replace loss function (e.g. with AM-Softmax)
2) Add another classifier model (CNN-based, RNN-based, Autoencoder etc.)
3) Performance evaluation and comparison between different models
