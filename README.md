# vp-openl3
This is voiceprint test project based on OpenL3 and TensorFlow.

1) Specify the path to the speech dataset and run embeddings.py to get embeddings from OpenL3 model.
2) Run train.py to train MLP-based speaker classifier using OpenL3 audio embeddings.
3) Run inference.py to extract MLP-based representations of embeddings and estimate EER using cosin similarity.
