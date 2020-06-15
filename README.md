# The Omniglot Challenge [(paper)](https://arxiv.org/abs/1902.03477)

Own Model for the Omniglot Challenge.

**Brain & Cognitive Society, IIT Kanpur**

## Proposed methodology
1.  Convert all stroke data to 25-point splines. :white_check_mark:
2.  Generate a **b**-vector using a variational autoencoder for each and every such stroke. **(In progress)**
3.  Use clustering to get number of primitives, and then turn them into vectors for each and every image in the background set using one-hot encoding.
4.  Perform supervised learning using a Convolutional Neural Network on the images and respective vectors.
5.  Then use the trained model for classification or one-shot learning tasks by introducing another model which maps the latent space of the image vector to desired output.
6.  Report acquired results.
