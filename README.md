# Lunar_Landscape_Detection
This is our project for our deep learning course at Centrale Paris, France.

To train the model or visualize the database

Add the following environment variables to a file named config.py at the root of the directory:\
- DATAPATH : path to the directory artificial-lunar-rocky-landscape-dataset like this : DATAPATH = "/mypath/artificial-lunar-rocky-landscape-dataset/" that you have previously downloaded from www.kaggle.com/romainpessia/artificial-lunar-rocky-landscape-dataset/data#\
- KERASPATH : path to keras_pretrained_models that you should have downloaded from https://www.kaggle.com/gaborfodor/keras-pretrained-models#vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5

Create and activate the environment from Lunar_Landscape_Detection.yml:\
```conda env create -f Lunar_Landscape_Detection.yml```\
```conda activate Lunar_Landscape_Detection```

Then you can run the app, providing a appropriate conda env. 
 - ```python Lunar_Landscape_Detection display``` :  allows you to visualize the image collection
 - ```python Lunar_Landscape_Detection``` :  trains the model. The output will be found in 
 - ```python Lunar_Landscape_Detection test #imagenumber``` :  applies the neural network to the image number #imagenumber 
