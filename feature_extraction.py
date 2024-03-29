from os import listdir
from pickle import dump
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
import string
from timeit import default_timer as timer

start = timer()

# extract features from each photo in the directory
def extract_features(directory):
    # load the model
    model = InceptionV3()
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # summarize
    print(model.summary())
    # extract features from each photo
    features = dict()
    i = 0
    for name in listdir(directory):
        # load an image from file
        i+=1
        print("image number : ",i)
        filename = directory + '/' + name
        image = load_img(filename, target_size=(299, 299))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get features
        feature = model.predict(image, verbose=0)
        # get image id
        image_id = name.split('.')[0]
        # store feature
        features[image_id] = feature
        print(name)
    return features
 
# extract features from all images
directory = 'images directory'
features = extract_features(directory)
print('Extracted Features: ', len(features))
# save to file
dump(features, open('features.pkl', 'wb'))

print(timer() - start)
