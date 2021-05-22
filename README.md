# -nofakes-project
this is the official repository for the #nofakes project provided by the University of Warwick AI society. We are building up on the classifier created by the adobe research team: https://github.com/PeterWang512/FALdetector. The aim of this project is to detect photoshopped images done to the body, not just human faces. The Train_YoloV3.ipynb allows us to train the yolo algorithm on custom labelled images to localize a human body. The model_trial.py in the networks folder uses a dilated residual network from the FAL repository, it is trained as a binary classifier to recognise whether an image is photoshopped or not. If it has been photoshopped then the yolo algorithm will be able to localize the body. All the necessary code has been uploaded to this GitHub repository but please see the references.docx for the references.

# Main results:
The images below display the predicted and ground truth flow values respectively. The predicted flow values are discretised and can take 121 different values. At the moment the model is not fine tuned with regression but the predictions seem consistent with the training data. 
<p float="left">
  <img src="https://github.com/peterfazekas1999/-nofakes-project/blob/main/results_trial/ground_truth.jpg" width ="300">
  <img src="https://github.com/peterfazekas1999/-nofakes-project/blob/main/results_trial/prediction1.jpg" width ="300">
</p>

<p float="left">
<img src="https://github.com/peterfazekas1999/-nofakes-project/blob/main/results_trial/ground_truth2.jpg" width ="300">
<img src="https://github.com/peterfazekas1999/-nofakes-project/blob/main/results_trial/prediction2.jpg" width ="300">
</p>

