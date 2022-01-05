# \#nofakes-project
This is the (still in construction!) official repository for the #nofakes project organized by the University of Warwick AI society. We are building up on the [classifier created by an Adobe research team](https://github.com/PeterWang512/FALdetector). The aim of this project is to detect photoshop editing on the whole human body, instead of just faces. The `yoloV3/Train_YoloV3.ipynb` file allows us to train the yolo algorithm on custom labelled images to localize a human body. The `networks/model_trial.py` file uses a dilated residual network from the FAL repository, which is trained as a binary classifier to recognise whether an image is photoshopped or not. If it has been photoshopped then we use the YOLO algorithm to find the bodies, and cut them out the picture, before feeding each body image to a model that predicts which pieces of the body have been edited (this is done by predicting the optical flow between the original picture and the edited picture. See [this paper](https://arxiv.org/abs/1906.05856) for more detail. The repository is still in construction, but most of the necessary code has been uploaded. See `references.docx` for the references.
# Project members 
Peter Fazekas, Nikhil Khetani, Paul Lezeau, Sanjif Shan. 
# Model training:
The most up to date code for the model training is in the `flow_training_22_05_2021.ipynb` file which trains the 121-perpixel classifier as well as the regression model to fine tune the flow predictions. This is a google colab notebook and the model was trained on a cloud GPU.
# Main results:
The images below display the predicted and ground truth flow values respectively. This model is the most up to date one and it incorporates regression fine tuning. Initially the model is trained as a 121 multiclass classifier to predict a set of discretized flow values. We then fine-tune this model by loading the learned model weights into a new model which then uses regression and a different loss function to fine-tune the flow predictions. The results of the final model are shown below
<p float="left">
  <img src="https://github.com/peterfazekas1999/-nofakes-project/blob/main/results_trial/ground_truth.jpg" width ="300">
  <img src="https://github.com/peterfazekas1999/-nofakes-project/blob/main/results_trial/prediction1.jpg" width ="300">
</p>

<p float="left">
<img src="https://github.com/peterfazekas1999/-nofakes-project/blob/main/results_trial/ground_truth2.jpg" width ="300">
<img src="https://github.com/peterfazekas1999/-nofakes-project/blob/main/results_trial/prediction2.jpg" width ="300">
</p>

# TODO
- Clean up and optimise code
- Make code self contained (no external files references) and easier to use.

# Disclaimer
`train.csv` and all the files that appear in folders `FALdetector_files` and `weights` are from [Adobe team repository](https://github.com/PeterWang512/FALdetector). 

