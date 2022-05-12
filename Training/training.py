import numpy as np
import logging, os
from train_model import *
from tensorflow.keras.callbacks import EarlyStopping

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

classes=["COVID_19 +ve","COVID_19 -ve"]
img_size=224
no_of_epochs=30


train_images_path="C:\\Users\\musel\\OneDrive\\Desktop\\FYPGeorgeMuselin\\dataset\\train_images.npy"
train_labels_path="C:\\Users\\musel\\OneDrive\\Desktop\\FYPGeorgeMuselin\\dataset\\train_labels.npy"


valid_images_path="C:\\Users\\musel\\OneDrive\\Desktop\\FYPGeorgeMuselin\\dataset\\valid_images.npy"
valid_labels_path="C:\\Users\\musel\\OneDrive\\Desktop\\FYPGeorgeMuselin\\dataset\\valid_labels.npy"

model_path="C:\\Users\\musel\\OneDrive\\Desktop\\FYPGeorgeMuselin\\dataset\\"


train_images=np.load(train_images_path)
train_labels=np.load(train_labels_path)


valid_images=np.load(valid_images_path)
valid_labels=np.load(valid_labels_path)


early_stop=EarlyStopping(monitor='val_loss',patience=10,verbose=0, 
                         mode='min',restore_best_weights=True)


densenet=train_model(model_path,train_images,train_labels,
                     valid_images,valid_labels,model_name="densenet201",
                     epochs=no_of_epochs,input_shape=(img_size,img_size,3),
                     classes=len(classes),callbacks=[early_stop])


inception=train_model(model_path,train_images,train_labels,
                      valid_images,valid_labels,model_name="inception_v3",
                      epochs=no_of_epochs,input_shape=(img_size,img_size,3),
                      classes=len(classes),callbacks=[early_stop])


resnet=train_model(model_path,train_images,train_labels,
                   valid_images,valid_labels,model_name="resnet50_v2",
                   epochs=no_of_epochs,input_shape=(img_size,img_size,3),
                   classes=len(classes),callbacks=[early_stop])

