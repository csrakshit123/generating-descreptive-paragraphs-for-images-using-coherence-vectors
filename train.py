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
            if input.size() != torch.Size([1,1,256]):
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
 
global USE_CUDA # if to use CUDA
USE_CUDA = False
global MAX_SENTC # max number of sentences in the paragraphs 
MAX_SENTC = 7
global L_S # learning rate
L_S = 5.0
global L_W # learning rate
L_W = 1.0
global  lamb # alpha value in coherencce equation
lamb = 0.6

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return load(f)

paragraphs = load_obj('files/generated_files/description_dict')

pickle_filename = 'files/features.pkl'

with open(pickle_filename,'rb') as f:
    features = load(f) #load the features dictionary from the pickle file

hidden_size_p = 256
output_size_p = 7047 # vocab size
vec_size_p = 256
coher_hidden_size_p = hidden_size_p
topic_hidden_size_p = hidden_size_p 
# all hidden sizes are same
nos_imgfeat_p = 2048 # feature vector
cont_flag_p = 1
n_layers_cont_p = 1 
n_layers_text_p = 1 
n_layers_couple_p = 1 
# criterion_1 = nn.CrossEntropyLoss() # loss function 1
criterion_1 = nn.MSELoss()
criterion_2 = nn.CrossEntropyLoss() # loss function 2
####### OUTPUT_SIZE has been used as per the vocabulary size for 5000 images . Change it according to number of images from preprocessing. #########

model = im2p(hidden_size = hidden_size_p, output_size = output_size_p, vec_size = vec_size_p, coher_hidden_size = coher_hidden_size_p, topic_hidden_size = topic_hidden_size_p, nos_imgfeat = nos_imgfeat_p, cont_flag = cont_flag_p, n_layers_cont = n_layers_cont_p, n_layers_text = n_layers_text_p, n_layers_couple = n_layers_couple_p) # arguments to be passed
# optimizer
optimizer = optim.Adam(model.parameters())
torch.autograd.set_detect_anomaly(True)

#training starts
for epochs in range(0,10):
    k = 0
    loss_epoch_list = []
    epoch_timer = timer()
    for img_id,p in paragraphs.items():
        k=k+1
        img_timer = timer()
        print("epoch = ",epochs,end = ' ')
        print("image id : ",img_id,end = ' ')
        print("image_number : ",k)
        optimizer.zero_grad()
        loss = 0
        feats = features[img_id]
        input_variable = p
        target_variable = input_variable # target variable to compare with the output to find loss
        model_hidden_st = None # Stores the hidden state vector at every step of the Sentence RNN
        nos_sentc = len(input_variable); sent_exec = 0
        sent_cand = 0
        for st in range(MAX_SENTC): # Iterate to see how many sentences the model intends to generate
            
            temp_ip = torch.from_numpy(feats)
            temp_ip = temp_ip.float()
            mod_ip = Variable(temp_ip, requires_grad=True) # Push in the Image Feature Here

            if st == 0:
                temp_hid = np.zeros(hidden_size_p, dtype = np.float32) # random.uniform(0, 1, (hidden_size - star_embed ) )
                temp_hid = temp_hid.reshape(1, 1, hidden_size_p)
                model_hidden = Variable(torch.from_numpy(temp_hid), requires_grad=True)
            else:
                mh = model_hidden_st.cpu().data.detach().numpy()
                model_hidden =  Variable(torch.from_numpy( mh[0, 0, :hidden_size_p].reshape(1, 1, hidden_size_p) ), requires_grad=True)

            # Check if Variable should be moved to GPU
            if USE_CUDA:
                mod_ip = mod_ip.cuda()
                model_hidden = model_hidden.cuda()

            # Call the model for the first time at the beginning of a sentence
            output_contstop, model_hidden = model(mod_ip, model_hidden, 'level_1') # Indicating that the first level RNN is to be used
            model_hidden_st = model_hidden
            strtstp_topv, strtstp_topi = output_contstop.data.topk(1)
            strtstp_ni = strtstp_topi[0][0]

            if strtstp_ni == 0: # So we continue
                sent_cand = sent_cand + 1
        sent_cand_temp = torch.tensor(float(sent_cand))#,requires_grad = True)
        nos_sentc_temp = torch.tensor(float(nos_sentc))#,requires_grad = True)
        loss = loss + L_S * criterion_1(sent_cand_temp,nos_sentc_temp) # The cross-entropy loss over the number of sentences
      
        val_sent = nos_sentc
        # Create the array of topic vectors and construct the Global Topic Vector - Topic Generation Net
        gl_mh = np.zeros((1, 1, hidden_size_p, val_sent))
        model_hidden_st = None
        # Stack up the vectors
        for st in range(nos_sentc): # Iterate over each sentence separately
            if len(input_variable[st]) <= 1: # If the sentence is of unit length, skip it
                continue
            
            temp_ip = torch.from_numpy(feats)
            temp_ip = temp_ip.float()
            mod_ip = Variable(temp_ip, requires_grad=True)

            if sent_exec == 0: # The first sentence
                temp_hid = np.zeros(hidden_size_p, dtype = np.float32) # random.uniform(0, 1, (hidden_size) )
                temp_hid = temp_hid.reshape(1, 1, hidden_size_p)
                model_hidden = Variable(torch.from_numpy(temp_hid), requires_grad=True) # Push in the Image Feature Here #encoder_hidden
                sent_exec = sent_exec + 1
            else: # All other sentences are initialized from previous sentences
                mh = model_hidden_st.cpu().data.detach().numpy()
                model_hidden =  Variable(torch.from_numpy( mh[0, 0, :hidden_size_p].reshape(1, 1, hidden_size_p) ), requires_grad=True) # Obtain the hidden state from the previous hidden state
                sent_exec = sent_exec + 1

            # Check if Variable should be moved to GPU
            if USE_CUDA:
                mod_ip = mod_ip.cuda()
                model_hidden = model_hidden.cuda()

            output_contstop, model_hidden = model(mod_ip, model_hidden, 'level_1') # level_1 indicates that we are using the Senetence RNN
            model_hidden_st = model_hidden
            gl_mh[0, 0, :, sent_exec-1] = (model(model_hidden_st, None, 'topic')[0].cpu().data.detach().numpy()).reshape(1, 1, hidden_size_p) # Transform the hidden state to obtain the topic vector
        
        # Compute the global topic vector as a weighted average of the individual topic vectors
        glob_vec = gl_mh[0, 0, :, 0].reshape(1, 1, hidden_size_p)
        for i in range(1, val_sent):
            glob_vec[:, :, :] = glob_vec[:, :, :].copy() + gl_mh[:, :, :, i].reshape(1, 1, hidden_size_p) * (LA.norm(gl_mh[:, :, :, i].reshape(-1)) / np.sum(LA.norm(gl_mh[:, :, :, :].reshape(-1, val_sent).T, axis=1)))


        # Process the Sentence RNN
        #Previous Hidden State Vector - The Coherence Vector
        prev_vec = ( np.zeros((1, 1, hidden_size_p)) ).astype(np.float32)

        for st in range(nos_sentc): # Iterate over each sentence separately
            
            if len(input_variable[st]) <= 1: # If the sentence is of unit length, skip it
                continue
            ip_var = Variable(torch.LongTensor(input_variable[st]))#, requires_grad=True) # One sentence
            op_var = Variable(torch.LongTensor(target_variable[st]))#, requires_grad=True)
            input_length = ip_var.size()[0]
            target_length = op_var.size()[0]
            
            loc_vec = (gl_mh[:, :, :, st]).reshape(1, 1, -1) # The original topic vector for the current sentence
            comb = np.add((1 - lamb) * loc_vec[0, 0, :], (lamb) * prev_vec[0, 0, :]) # Combine the current topic vector and the coherence vector from the previous sentence
            glob_vec = glob_vec.astype(np.float32)
            if type(comb) is not np.ndarray:
                foo = comb.numpy()
                comb = foo
            comb = comb.astype(np.float32)
            mh = (((model(torch.tensor([[glob_vec[0, 0,:]]]), torch.tensor([[comb]]), 'couple' )[0]).reshape(1, 1, -1)).detach().numpy()).astype(np.float32) # Coupling Unit
            
            # Construct the input for the first word of a sentence in the Sentence RNN
            model_input =  Variable(torch.from_numpy(mh[0, 0, :]), requires_grad=True).reshape(1,1,256)
            model_hidden = Variable(torch.from_numpy(temp_hid), requires_grad=True).reshape(1,1,256)

            #print("model_input",model_input)
            if USE_CUDA:
                model_hidden = model_hidden.cuda()
                model_input = model_input.cuda()
                ip_var = ip_var.cuda()
                op_var = op_var.cuda()
                
            # Teacher forcing: Feed the target as the next input
            for di in range(1, target_length, 1):
                model_output, model_hidden = model(model_input, model_hidden, 'level_2') # level_2 indicates that we want to use the Sentence RNN
                foo = loss
                loss = foo + L_W * criterion_2(model_output, op_var[di:di+1]) # Use the second cross-entropy term
                
                model_input = op_var[di:di + 1]
                			
            # Re-initialize the previous vector
            prev_vec = model(model_hidden, None, 'coher')[0].detach()
           
        # optimizer to be added	
        print("loss = ",loss,end = ' ')
#         loss_list[epochs].append(loss.data.item())
        loss.backward()
        optimizer.step()
        loss_epoch_list.append(loss.data.item())
        print("time taken for this image =",timer() - img_timer)
        print("this epoch =",timer() - epoch_timer)
        
    filepath = 'files/models/model' + str(epochs) + '.pth'
    torch.save(model,filepath)
    
    filename = 'files/losses/loss_epoch_' + str(epochs) + '.json'
    with open(filename,'w') as f:
        json.dump(loss_epoch_list,f)
