from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from timeit import default_timer as timer
import pickle
import json
import string
start = timer()

def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def load_descriptions(doc,dataset):
    mapping = dict()
    # process lines
    doc = doc.split('\n')
    doc.pop(0)
    doc = '\n'.join(doc)
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split(';')
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id , image_desc = tokens[1] , tokens[2:]
        if(int(image_id) in dataset):
        # convert description tokens back to string
             image_desc = ' '.join(image_desc)
        # create the list if needed
        # store description
             mapping[image_id] = image_desc.split('.')
             mapping[image_id].pop()
    return mapping

def clean_descriptions(descriptions):
    # prepare translation table for removing punctuation
    x = list(string.punctuation)
    x.remove(',')
    x = ''.join(x)
    
    table = str.maketrans('', '', x)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word)>1 or (',' in word)]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha() or (',' in word) ] 
            # store as string
            desc_list[i] =  ' '.join(desc)

def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
    # load document
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if int(image_id) in dataset:
            # create list
            if image_id not in descriptions:
                descriptions[image_id] = list()
            # wrap description in tokens
            desc = ' '.join(image_desc)
            # store
            descriptions[image_id].append(desc)
    return descriptions

def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def create_sequences(tokenizer, descriptions):
    new_desc_list = dict()
    # walk through each image identifier
    for key, desc_list in descriptions.items():
        if(key not in new_desc_list.items()):
            new_desc_list[key] = list()
        # walk through each description for the image
        for desc in desc_list:
            # encode the sequence
            temp = tokenizer.texts_to_sequences([desc])[0]
            new_desc_list[key].append(temp)
    return new_desc_list

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

csv_filename = 'files/dataset.csv'
# load descriptions
doc = load_doc(csv_filename)


# if working on a smaller dataset run this code
# with open('train_split.json',"r") as f:
#     k = f.read()
#     x = json.loads(k)
# x = x[:1000]

# with open('train_split_1000.json','w') as f:
#     json.dump(x,f)




train_imgs = 'files/train_split.json'
with open(train_imgs) as f:
    k = f.read()
    train = json.loads(k)

# parse descriptions
descriptions = load_descriptions(doc,train)
print("loaded description",len(descriptions))

# clean descriptions
clean_descriptions(descriptions)
# save to file
desc_filename = 'files/generated_files/descriptions.txt'
save_descriptions(descriptions, desc_filename)


# descriptions

train_descriptions = load_clean_descriptions(desc_filename, train)
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
wrd_list = tokenizer.word_index
wrd_list = dict([(value, key) for key, value in wrd_list.items()]) 
print("Vocab size = ",vocab_size)
description_dict = create_sequences(tokenizer, train_descriptions)
desc_dict_filename = 'files/generated_files/description_dict'
save_obj(description_dict, desc_dict_filename)

word_list_file = 'files/generated_files/word_list.json'
with open(word_list_file,'w') as f:
    json.dump(wrd_list,f)
