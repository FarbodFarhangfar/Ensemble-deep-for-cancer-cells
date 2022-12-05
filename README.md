# Ensemble
Esnemble deep learning for cancer cell detection


We started with the idea of cancer cell classification with deep neural networks.
There are many classification techniques but we focused on ensemble classification methods and get a better result from all of them combined.
Dataset is a breast cancer image set with tiff format that was provided as a Kaggle.com challenge
We also used a new preprocessing method that can find hotspots according to our needs.
For preprocessing we cut images in 224*224 cubes so we can find hotspots in an image, normalize them and use them as our data.
Images are large and we don't need most of the image, also there are white spots that we need to cut out.
We feed these samples to pre-trained models so we get raw forecast tensors.
We feed these tensors to a deep cluster. As a result, we get clustered data and we feed them to parallel deep convolutional neural networks. Then we concatenate the results with a multimodal architecture.
Then aggregate them with modality weight predictions
At this level, we have a focused tensor that points to the part that we need to feed to our ensemble members.
We used up-to-date classification methods (Dennet, condensate, …) as our ensemble members and train our data separately on each one of them.
Then feed their result to a probability distribution ensemble that produces an output and a probability using a base learner (we used VGG).
results can be fed to the CRPS loss function that acts as a natural gradient and trains the model. The result
we also used Bayesian optimization for hyperparameter tuning.
I write the code mainly with the Pytorch framework and other Python libraries.
many articles and GitHub-provided codes for classification have been used.
The paper is in the final stages of publishing.


Farbod Farhangfar.
