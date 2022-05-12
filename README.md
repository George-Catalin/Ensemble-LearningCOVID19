# Ensemble-LearningCOVID19
This repository contains all the scripts required for data preparation, model training, and the development of the GUI App.
## About
COVID-19 continues to have a devastating impact on people’s lives all across the
world. To combat this condition, it is critical to test affected patients in a quick, lowcost, and effective manner. Radiological examination, with Chest X-ray being the most readily available and least priced alternative, is one of the most promising stages toward accomplishing this goal. A Deep Convolutional Neural Network (CNN)-based
method is presented in this project report to help detect COVID-19 positive patients utilising Chest X-Ray images. The proposed study uses many state-of-the-art CNN models that are well-known for image classification tasks, including DenseNet201, Resnet50V2, and InceptionV3. All have been independently trained to make independent forecasts. The models are then merged to predict a category value using a new weighted average ensemble technique. Publicly available Chest X-ray images of
COVID positive and COVID negative cases are used to test the efficacy of the solution. COVID-19 is a new condition, so there is not a lot of information about it. The sorted dataset gathered from Kaggle and used, contains 5200 images of COVID-19 positive patients and COVID-19 negative subjects. The suggested ensemble model outperformed state-of-the-art CNN models with a classification accuracy of 97.33%. In addition, a web-based graphical user interface (GUI) application for public use was created and also Grad-CAM visualization was used to draw attention to key areas in the image where the predictions were made. The web-based tool can be utilised by
any medical or non-medical professionals on any computer/device to detect COVID positive patients leveraging Chest X-Ray images in seconds.

## Dependencies 
*  Python 3.7.11
*  Tensorflow 2.8.0
*  Flask
*  Numpy
*  Pandas
*  Scikit-Learn

## Strcture
The project has four directories:
*  DataPreparation Script : Contains NPY_creator.py script for making npy arrays out of the images
*  Model Training Script :
  * train_model.py
