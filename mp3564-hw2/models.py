"""
COMS 4705 Natural Language Processing Fall 2019
Kathy McKeown
Homework 2: Emotion Classification with Neural Networks - Models File
Authors: Elsbeth Turcan <eturcan@cs.columbia.edu>

<Malavika Pande>
<mp3564>
I am using 2 late days for this assignment 
"""

import torch.nn as nn
import torch.nn.utils.rnn as rnn
import utils
import pickle 
import torch.nn.functional as F
TEMP_FILE = "temporary_data.pkl" 
import torch.tensor

class DenseNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DenseNetwork, self).__init__()
        ########## YOUR CODE HERE ##########
        # TODO: Here, create any layers and attributes your network needs.
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Linear Layers 
        self.linear1 = nn.Linear(input_dim, 84)
        self.linear2 = nn.Linear(84, output_dim)
    
        # Load pretrained embeddings  
        with open(TEMP_FILE,"rb") as f:
                print("Loading DataLoaders and embeddings from file....")
                pretrained_embeds = pickle.load(f)[3]
        
        self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embeds)   

        return
        raise NotImplementedError

    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # TODO: Fill in the forward pass of your neural network.
        # TODO: (The backward pass will be performed by PyTorch magic for you!)
        # TODO: Your architecture should...
        # TODO: 1) Put the words through an Embedding layer (which was initialized with the pretrained embeddings);  
        # TODO: 2) Take the sum of all word embeddings in a sentence; and
        # TODO: 3) Feed the result into 2-layer feedforward network which produces a 4-vector of values,
        # TODO: one for each class

        x = self.word_embeddings(x) 
        print(x.shape)
        x = torch.sum(x, dim=1)
        print(x.shape)
        # Define how data flows through the network 
        x = F.tanh(self.linear1(x.float()))
        print(x.shape)
        x = F.tanh(self.linear2(x))
        print(x.shape)

        return x 

        raise NotImplementedError

# https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
class RecurrentNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(RecurrentNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        # TODO: Here, create any layers and attributes your network needs.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim 
        
        # RNN and Linear Layers
        # Number of layers is 2 
        self.rnn = nn.RNN(input_dim, hidden_dim, 2, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

        with open(TEMP_FILE,"rb") as f:
            print("Loading DataLoaders and embeddings from file....")
            pretrained_embeds = pickle.load(f)[3]
        
        self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embeds)
        
        return 
        raise NotImplementedError

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # TODO: Fill in the forward pass of your neural network.
        # TODO: (The backward pass will be performed by PyTorch magic for you!)
        # TODO: Your architecture should...
        # TODO: 1) Put the words through an Embedding layer (which was initialized with the pretrained embeddings);
        # TODO: 2) Feed the sequence of embeddings through a 2-layer RNN; and
        # TODO: 3) Feed the last output state into a dense layer to become a 4-vector of values, one for each class

        x = self.word_embeddings(x)
        x = x.float()
        batch_size = x.size(0)

        # Initialize hidden layer with zeroes 
        init_hidden = self.get_init_hidden(batch_size)

        out, hx = self.rnn(x, init_hidden)
        
        out = self.linear(out[:, -1, :])
        
        
        return out
        
        raise NotImplementedError
    # Added this function because was not getting correct hidden dimensions before
    # Need to update  
    def get_init_hidden(self, batch_size):
        return torch.zeros(2, batch_size, self.hidden_dim)

# TODO: If you do any extensions that require you to change your models, make a copy and change it here instead.
# TODO: PyTorch unfortunately requires us to have your original class definitions in order to load your saved
# TODO: dense and recurrent models in order to grade their performance.
class ExperimentalNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ExperimentalNetwork, self).__init__()
        ########## YOUR CODE HERE ##########
        # change architecture settings 
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Linear Layers 
        self.linear1 = nn.Linear(input_dim, 84)
        self.linear2 = nn.Linear(84, output_dim)
    
        # Load pretrained embeddings  
        # I altered pre-processing code 
        with open(TEMP_FILE,"rb") as f:
                print("Loading DataLoaders and embeddings from file....")
                pretrained_embeds = pickle.load(f)[3]
        
        self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embeds)   

        return
        raise NotImplementedError
    
    # x is a padded sequence for RNN
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        
        x = self.word_embeddings(x) #[128, 82, 100]  [batch_size, seq length, embedding dim]
        print(x.shape) #
        # extension-grading
        x = torch.mean(x, dim=1) # average over word sequence dimension [128, 100] 
        print(x.shape)
        
        x = F.tanh(self.linear1(x.float()))
        x = F.tanh(self.linear2(x))

        return x 

        raise NotImplementedError
        
    
    