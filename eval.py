import json
from pickle import load
import numpy as np
import pandas as pd
from numpy import linalg as LA
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from timeit import default_timer as timer
import sys
from nltk.translate.bleu_score import sentence_bleu

global USE_CUDA # if to use CUDA
USE_CUDA = False
global MAX_SENTC # max number of sentences in the paragraphs 
MAX_SENTC = 8
global L_S # learning rate
L_S = 5.0
global L_W # learning rate
L_W = 1.0
global  lamb # alpha value in coherencce equation
lamb = 0.6
global max_length
max_length = 10 # to set the length

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

csv_filename = 'final/files/dataset.csv'
doc = load_doc(csv_filename)

pickle_filename = 'final/files/features_test.pkl'
f = open(pickle_filename,'rb')
features = load(f) #load the features dictionary from the pickle file
f.close()

test_imgs = 'final/files/generated_files/test_split_1000.json'
with open(test_imgs) as f:
    k = f.read()
    test = json.loads(k)

descriptions = load_descriptions(doc,test)


class im2p(nn.Module):
    def __init__(self, hidden_size, output_size, vec_size, coher_hidden_size, topic_hidden_size, nos_imgfeat, cont_flag, n_layers_cont, n_layers_text, n_layers_couple):
        super(im2p,self).__init__()
        self.n_layers_cont = n_layers_cont
        self.n_layers_text = n_layers_text
        self.n_layers_couple = n_layers_couple
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vec_size = vec_size
        self.coher_hidden_size = coher_hidden_size
        self.topic_hidden_size = topic_hidden_size
        self.nos_imgfeat = nos_imgfeat
        self.cont_flag = cont_flag
        # continue stop network
        self.img_encoding = nn.Linear(nos_imgfeat, hidden_size)
        self.embedding = nn.Embedding(output_size, hidden_size) # For handling the text inputs
        self.gru_cont = nn.GRU(hidden_size, hidden_size, n_layers_cont) # GRU for start stop
        self.gru_text = nn.GRU(hidden_size, hidden_size, n_layers_text) # GRU for sentence
        self.out_cont = nn.Linear(hidden_size, cont_flag) # Flag indicating if we should continue
        self.out_text = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
        # coupling network
        self.gru_couple = nn.GRU(vec_size, hidden_size, n_layers_couple) # GRU for the coupling unit
        # Coherence Network
        self.fc_1_coher = nn.Linear(hidden_size, coher_hidden_size) # First Layer
        self.fc_2_coher = nn.Linear(coher_hidden_size, hidden_size) # Second Layer
        self.non_lin_coher = nn.SELU(inplace = False)
        # Topic Network
        self.fc_1_topic = nn.Linear(hidden_size, topic_hidden_size) # First Layer
        self.fc_2_topic = nn.Linear(topic_hidden_size, hidden_size) # Second Layer
        self.non_lin_topic = nn.SELU(inplace = False)

    def forward(self, input, hidden, flag):
        if flag == 'level_1': # Passing image features and stars for the first GRU of every sentence - Sentence RNN
            ip = self.img_encoding(input) # .view(1, 1, -1)
            ip = ip.view(1, 1, -1)
            output, hidden = self.gru_cont(ip, hidden)
            k = self.out_cont(output[0])
            output = self.softmax(k) # Obtain the labels of whether to continue or stop
        elif flag == 'level_2': # Passing word embeddings - Word RNN
            output = input
            if input.size() != torch.Size([1,1,512]):
                output = self.embedding(output).view(1, 1, -1)
                output = F.relu(output)
                output, hidden = self.gru_text(output, hidden)
            output, hidden = self.gru_text(output, hidden)
            output = self.out_text(output[0])
            self.softmax = nn.LogSoftmax()
            output = self.softmax(output)	
        elif flag == 'couple': # Forward through the coupling unit
            output, hidden = self.gru_couple(input, hidden)
        elif flag == 'coher': # Forward through the Coherence Vector Network
            output = self.fc_1_coher(input)
            output = self.non_lin_coher(output)
            output = self.fc_2_coher(output)
            output = self.non_lin_coher(output)
            hidden = None
        elif flag == 'topic': # Forward through the Coherence Vector Network
            output = self.fc_1_topic(input)
            output = self.non_lin_topic(output)
            output = self.fc_2_topic(output)
            output = self.non_lin_topic(output)
            hidden = None
        return output, hidden
model = torch.load('model6.pth')
hidden_size_p = 512
output_size_p = 7047 # vocab size
vec_size_p = 512
coher_hidden_size_p = hidden_size_p
topic_hidden_size_p = hidden_size_p 
# all hidden sizes are same
nos_imgfeat_p = 2048 # feature vector
cont_flag_p = 1
n_layers_cont_p = 1 
n_layers_text_p = 1 
n_layers_couple_p = 1 

with open('final/files/generated_files/word_list.json',"r") as f:
    k = f.read()
    wrd_list = json.loads(k)
wrd_list = dict([(int(key),value) for key, value in wrd_list.items()]) 
pickle_filename = 'final/files/features_test.pkl'
f = open(pickle_filename,'rb')
features = load(f) #load the features dictionary from the pickle file
f.close()
# img_id = '2406949'

all_scores1 = []
all_scores2 = []
all_scores3 = []
all_scores4 = []

image_no = 0
for img_id,reference in descriptions.items(): 
    model_hidden_st = None # Stores the hidden state vector at every step of the Sentence RNN
    pred_words = [] # Stores the list of synthesized words

    # Create the array of topic vectors and the Global Topic Vector -- Topic Generation Net
    gl_mh = np.zeros((1, 1, hidden_size_p, MAX_SENTC)); val_sent = 0;

    for st in range(MAX_SENTC): # Iterate over each sentence separately

        feats = features[img_id] 


        temp_ip = torch.from_numpy(feats)
        temp_ip = temp_ip.float()
        mod_ip = Variable(temp_ip) # Push in the Image Feature Here

        if st == 0: # Initialize the hidden state for the first se
            temp_hid = np.zeros(hidden_size_p, dtype = np.float32) # random.uniform(0, 1, (opt.hidden_size - star_embed ) )
            temp_hid = temp_hid.reshape(1, 1, hidden_size_p )
            model_hidden = Variable(torch.from_numpy(temp_hid))
        else:
            mh = model_hidden_st.cpu().data.numpy()
            model_hidden =  Variable(torch.from_numpy(mh[0, 0, :hidden_size_p].reshape(1, 1, hidden_size_p)))

        # Check if Variable should be moved to GPU
        if USE_CUDA:
            mod_ip = mod_ip.cuda()
            model_hidden = model_hidden.cuda()

        output_contstop, model_hidden = model(mod_ip,model_hidden,'level_1') # level_1 indicates that we are using the Senetence RNN
        model_hidden_st = model_hidden
        strtstp_topv, strtstp_topi = output_contstop.data.topk(1)
        strtstp_ni = strtstp_topi[0][0]

        if strtstp_ni == 0: # So we continue
            val_sent = val_sent + 1
            gl_mh[0, 0, :, st] = (model(model_hidden_st, None, 'topic')[0].cpu().data.numpy()).reshape(1, 1, hidden_size_p) # Transform the hidden state to obtain the topic vector

    # Compute the Global Topic Vector as a weighted average of the individual topic vectors
    glob_vec = gl_mh[0, 0, :, 0].reshape(1, 1, hidden_size_p)
    for i in range(1, val_sent):
        glob_vec[:, :, :] += gl_mh[:, :, :, i].reshape(1, 1, hidden_size_p) * (LA.norm(gl_mh[:, :, :, i].reshape(-1)) / np.sum(LA.norm(gl_mh[:, :, :, :].reshape(-1, val_sent).T, axis=1)))

    # Sentence Generation Net
    #Previous Hidden State Vector
    prev_vec = (np.zeros((1, 1, hidden_size_p))).astype(np.float32)

    for st in range(MAX_SENTC): # Iterate over each sentence separately and generate the words

        sentence = []
        loc_vec = (gl_mh[:, :, :, st]).reshape(1, 1, -1) # The original topic vector for the current sentence
        comb = np.add((1-lamb) * loc_vec[0, 0, :], (lamb) * prev_vec[0, 0, :]) # Combine the current topic vector and the coherence vector from the previous sentence
        if type(comb) is not np.ndarray:		
            foo = comb.numpy()
            comb = foo	
        comb = comb.astype(np.float32)
        glob_vec = glob_vec.astype(np.float32)	
        mh = (((model(torch.tensor([[glob_vec[0, 0,:]]]), torch.tensor([[comb]]), 'couple' )[0]).reshape(1, 1, -1)).detach().numpy()).astype(np.float32)	

        # Construct the input for the first word of a sentence in the Sentence RNN

        model_input =  Variable(torch.from_numpy(mh[0, 0, :]), requires_grad=True).reshape(1,1,hidden_size_p)
        model_hidden = Variable(torch.from_numpy(temp_hid), requires_grad=True).reshape(1,1,hidden_size_p)

        if USE_CUDA:
            model_hidden = model_hidden.cuda()
            model_input = model_input.cuda()

        for di in range(max_length):

            model_output, model_hidden = model(model_input, model_hidden, 'level_2') # level_2 indicates that we want to use the Sentence RNN
            topv, topi = model_output.data.topk(1) # Standard RNN decoding of the words
            ni = topi[0][0]
            ni = ni.data.item()
            # Check if EOS has been predicted
            if ni == len(wrd_list):
                sentence.append('<EOS>')
                break
            else:
                sentence.append(wrd_list[ni])
                model_input = Variable(torch.LongTensor([ni]))
        if sentence not in pred_words:
            pred_words.append(sentence)
    # Re-initialize the previous vector
        prev_vec = model(model_hidden, None, 'coher')[0].detach()
    candidate = pred_words 
    reference = descriptions[img_id]
	#reference = [i.split() for i in reference]
    score1 = 0
    score2 = 0
    score3 = 0
    score4 = 0

    for i in range(min(len(reference),len(candidate))):
        score1 += sentence_bleu([reference[i].strip().split()], candidate[i],weights=(1,0,0,0))
        score2 += sentence_bleu([reference[i].strip().split()], candidate[i],weights=(0,1,0,0))
        score3 += sentence_bleu([reference[i].strip().split()], candidate[i],weights=(0,0,1,0))
        score4 += sentence_bleu([reference[i].strip().split()], candidate[i],weights=(0,0,0,1))
    all_scores1.append(score1)
    all_scores2.append(score2)
    all_scores3.append(score3)
    all_scores4.append(score4)
    print("Image No.",image_no)
    image_no += 1

print("BLEU SCORE",sum(all_scores1)/len(all_scores1))
print("BLEU SCORE",sum(all_scores2)/len(all_scores2))
print("BLEU SCORE",sum(all_scores3)/len(all_scores3))
print("BLEU SCORE",sum(all_scores4)/len(all_scores4))

