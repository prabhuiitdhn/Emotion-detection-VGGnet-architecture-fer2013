This project contains the VGGnet architecture of Deep-learning in fer2013 dataset

The Fer2013:
The Kaggle Emotion and Facial Expression Recognition challenge training dataset consists of 28,709 images, each of which are 48×48 grayscale images (Figure 10.1). The faces have been
automatically aligned such that they are approximately the same size in each image. Given theseimages, our goal is to categorize the emotion expressed on each face into seven distinct classes:
angry, disgust, fear, happy, sad, surprise, and neutral.

This facial expression dataset is called the FER13 dataset and can be found at the official Kaggle
competition page and downloading the fer2013.tar.gz file:
http://pyimg.co/a2soy

The .tar.gz archive of the dataset is ≈ 92MB, so make sure you have a decent internet connection before downloading it. After downloading the dataset, you’ll find a file named fer2013.csv
with with three columns:
• emotion: The class label.
• pixels: A flattened list of 48×48 = 2;304 grayscale pixels representing the face itself.
• usage: Whether the image is for Training, PrivateTest (validation), or PublicTest
(testing).
Our goal is to now take this .csv file and convert it to HDF5 format so we can more easily
train a Convolutional Neural Network on top of it.


To run this program:

python emotion_detector.py --cascade haarcascade_frontalface_default.xml --model checkpoints/epoch_75.hdf5 --video prabhu.mp4
