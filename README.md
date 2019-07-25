# generating-descreptive-paragraphs-for-images-using-coherence-vectors

make sure these files are available:
feature_extraction.py
preprocessing.py
train.py
test.py
files/dataset.csv
files/train_split.json
fiels/test_split.json

dependancies requires : pytorch, Keras, tensorflow

Run the file feature_extraction.py 
inputs to this file : folder containing images
outputs of this file : files/feature.pkl

Run the file preprocessing.py 
inputs to this file : files/dataset.csv, files/train_split.json
outputs of this file : files/generated_files/descreption_dict.pkl, files/generated_files/descreptions.txt, files/generated_files/word_list.json

Run the file train.py
inputs to this file : files/feature.pkl, files/generated_files/descreption_dict.pkl, files/generated_files/word_list.json
outputs of this file : files/generated_files/models/model_x.pth, files/generated_files/losses/loss_x.json, x is the epoch number

Run the file test.py to predict for an image
inputs to this file : files/feature.pkl or files/feature_test.pkl based on the image you want to test.
feature.pkl has features for all train images
feature_test.pkl has features for all test images

Run the file eval.py to get the BLEU scores
inputs to this file : files/feature_test.pkl, files/test_split.json, files/generated_files/word_list.json, files/generated_files/models/mode_x.pth, x is the epoch number.

you can even run the notebook file.
