# -nofakes-project
This is the official repository for the #nofakes project organized by the University of Warwick AI society. We are building up on the classifier created by the adobe research team: https://github.com/PeterWang512/FALdetector. The aim of this project is to detect photoshopped images done to the body, not just human faces. The `yoloV3/Train_YoloV3.ipynb` file allows us to train the yolo algorithm on custom labelled images to localize a human body. The `networks/model_trial.py` file uses a dilated residual network from the FAL repository, it is trained as a binary classifier to recognise whether an image is photoshopped or not. If it has been photoshopped then the yolo algorithm will be able to localize the body. All the necessary code has been uploaded to this GitHub repository but please see the references.docx for the references.
# Project members 
Peter Fazekas, Nikhil Khetani, Paul Lezeau, Sanjif Shan. 
# Model training:
The most up to date code for the model training is the flow_training_22_05_2021.ipynb which trains the 121-perpixel classifier as well as the regression model to fine tune the flow predictions. This is a google colab notebook and the model was trained on a cloud GPU.
# Main results:
The images below display the predicted and ground truth flow values respectively. This model is the most up to date one and it incorporates regression fine tuning. Initially the model is trained as a 121 multiclass classifier to predict a set of discretized flow values. We then fine tune this model by loading the learned model weights into a new model which then uses regression to fine tune the flow predictions. The results of the final model are shown below
<p float="left">
  <img src="https://github.com/peterfazekas1999/-nofakes-project/blob/main/results_trial/ground_truth.jpg" width ="300">
  <img src="https://github.com/peterfazekas1999/-nofakes-project/blob/main/results_trial/prediction1.jpg" width ="300">
</p>

<p float="left">
<img src="https://github.com/peterfazekas1999/-nofakes-project/blob/main/results_trial/ground_truth2.jpg" width ="300">
<img src="https://github.com/peterfazekas1999/-nofakes-project/blob/main/results_trial/prediction2.jpg" width ="300">
</p>.

# Disclaimer
This repository is still being cleaned up, and a the minute a lot of the code that appears here was taken directly from the Adobe team repository https://github.com/PeterWang512/FALdetector. We of course take no credit for the work done by the Adobe Team. 

