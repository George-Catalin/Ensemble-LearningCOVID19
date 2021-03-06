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
   * train_model.py - Script for training the models
   * testing.py - Script for getting the performance metrics
   * training.py - Contains functions for ensembling and function for measuring the performance of the ensembler
   
   
*  GUI :
   * gradcam.py - Script for the GRADCAM implementation
   * main.py - Main application written in python using Flask
   * utils.py - Flask implementation to run the app on a browser
    
 ![alt text](https://github.com/George-Catalin/Ensemble-LearningCOVID19/blob/main/Pictures/result.jpg?raw=true)
 
 # Data Preparation
 ```
1. The Image data are kept into separate directories as COVID_19 +ve and COVID_19 -ve.
2. These images are split into separate diretories as train and test.

Thus the directory structure is as:

Images
    |
    ----Train
    |       |
    |       ---- COVID_19 +ve
    |       |
    |       ---- COVID_19 -ve
    |
    -----Test
            |
            ---- COVID_19 +ve
            |
            ---- COVID_19 -ve
    
3. Then run the NPY_creator.py script and the enter the paths according to the prompt.
 ```
#  Training the Model
```
1. Run the train_model.py script and the enter the paths that the program asks for.
2. The models will be saved in the directory.
```

#  Performance
```
After training the model, the accuracy, confusion matrix will be printed in the console.
```

#  The Application
```
Run the main.py script and copy and paste the link generated in the console on a browser.
Or simply see it on: http://localhost:5000/
```
