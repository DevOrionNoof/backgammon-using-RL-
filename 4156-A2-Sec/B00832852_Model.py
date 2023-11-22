#!/usr/bin/env python
# coding: utf-8

# # MLP Model Class

# In[72]:


# this class is for MLP
# first we initilize with random weigts with mean of 0 and std of 1
# we are using same biase for full layer 


# its def score() function returns returns score  of state  after forward propogation 
# its def TRAIN() function gets input data and label which is loss we got using TD algorithm equation then it calculates loss of that function
# if we want to use batch traning then we can update weigt after few samples traning in update_score() function


# In[87]:


import numpy as np
from backgammon import backgammon


#this class is MLP it have one hidden layer for traning it uses numpy matrix multiplication for forward and backward pass
#it also includes TD implementation in backward bass for learning
class Model:
    def __init__(self,input_shape,hidden_shape,output_shape,lr):
        #initilizing weights and biases
        np.random.seed(1101)
        self.lr=lr
        self.input_shape=input_shape
        self.hidden_shape=hidden_shape
        self.output_shape=output_shape
        self.w1=np.random.rand(hidden_shape,input_shape)-0.5
        self.w2=np.random.rand(output_shape,hidden_shape)-0.5
        self.b1=np.array(0.5)
        self.b2=np.array(0.5)
        self.grad1=np.zeros((output_shape,hidden_shape))
        self.grad2=np.zeros((hidden_shape,input_shape))
        self.gb1=np.zeros(hidden_shape)
        self.gb2=np.zeros(output_shape)
        
    def predict(self,data):
        i1=data.flatten()
        h1=np.matmul(self.w1,i1)+self.b1
        h1s = 1/(1 + np.exp(-h1))
        o=np.matmul(self.w2,h1s)+self.b2
        os=z = 1/(1 + np.exp(-o))
        return os
    def Train(self,data,output):   
        actual=np.array([[output]])
        i1=data.flatten()
        h1=np.matmul(self.w1,i1)+self.b1
        h1s = 1/(1 + np.exp(-h1))
        o=np.matmul(self.w2,h1s)+self.b2
        os=z = 1/(1 + np.exp(-o))
      
        sum1=(actual-os)
        loss=sum(np.square(sum1.copy()).flatten())
        s=(os*(1-os))*sum1

        s=s.reshape(self.output_shape,1)

        h1ss=h1s.reshape(1,self.hidden_shape)
        self.grad1+=np.matmul(s,h1ss)
        gs=self.grad1.sum(axis=0).reshape(self.hidden_shape,1)
        p=(h1s*(1-h1s)).reshape(self.hidden_shape,1)
        s=gs.mean()*p
        
        h1sss=i1.reshape(1,self.input_shape)

        self.grad2+=np.matmul(s,h1sss)

    def update_score(self):
        self.w1+=(self.grad2*self.lr)
        self.grad2=np.zeros((self.hidden_shape,self.input_shape))  
        self.w2+=(self.grad1*self.lr)
        self.grad1=np.zeros((self.output_shape,self.hidden_shape))


# # Agent CLass

# In[88]:


# this class have model object of MLP and its learning
# its preprocess_state() function normalize inputs
# return values function gets state and preprocess them and then return its value by pass it to MLP model
# its Train methode gets traning data and train  MLP model


# In[89]:


import numpy as np
import random

#this is agent class to train agent for playing game
class TDAgent:
    def __init__(self):
        
        self.lr = 0.003
        self.model = Model(16,250,1,self.lr)

    def preprocess_state(self, state):
        state=np.array(state)
        state= (state-state.mean())/state.std()
        return state
    
    def predict(self,state):
        state=self.preprocess_state(state)
        val=self.model.predict(state)
        return val
     
    def Train(self,state,val):
        state=self.preprocess_state(state)
        self.model.Train(state,val)
        self.model.update_score()        



# In[96]:
#
#
# #This was to train the model for either white player or black player by un comenting the player
#
# player='WHITE'
# #player='BLACK'
# if player=='WHITE':
#     sign=1
# else:
#     sign=-1
#
#
#
# # # Model Training
#
# # In[97]:
#
#
# # we use 1 reward for player we want to win
#
#
# # In[98]:
#
#
# game = backgammon()
# agent_model = TDAgent()
# total_accuracy,current_accuracy=[],[]
#
# #in traning loop i am using shape 32 as input which include current state concatinated with next state which gives better accuracy
#
# discount = 0.90
# lambda1= 0.9
#
# episodes=10000
#
# wins=0
# last_1000=[]
# for episode in range(episodes):
#     game = backgammon()
#     while game.get_winner() is None:
#         state= game.board
#         val=agent_model.predict(state)
#         moves=game.moves
#         scores=[]
#         for nn1 in moves:
#             scores.append(sign*(agent_model.predict(nn1)))
#         if moves[0][-1]==1 :
#             mov=np.argmax(scores)
#             state=moves[mov]
#             game.make_move(moves[mov])
#         else:
#             mov=np.argmin(scores)
#             state=moves[mov]
#             game.make_move(moves[mov])
#         next_state= game.board
#         next_val=agent_model.predict(next_state)
#
#         reward=0.0
#         if game.get_winner()==player:
#             reward=1.0
#         val=val+lambda1*(reward +discount*next_val-val)
#         agent_model.Train(state,val)
#
#
#         if game.get_winner()!=None:
#             break
#
#     if game.get_winner()==player:
#         wins+=1
#
#     last_1000.append(game.get_winner())
#
#
#
#     if episode%10==9:
#         last_1000=last_1000[-1000:]
#         total_accuracy.append((wins/(episode+1)))
#         current_accuracy.append(last_1000.count(player)/len(last_1000))
#         #break condition for traning as if we train to much it crashes
#         if len(current_accuracy)>4:
#             if current_accuracy[-1]>84 and current_accuracy[-2]>84:
#                 break
#         print(episode,": Total win percentage of  {}= ".format(player),int((wins/(episode+1))*100),"Last 100 Win Percentage of {}=".format(player),int(last_1000.count(player)/len(last_1000)*100))
#
#
# # In[99]:
#
#
# # this is to plot accuracy while traning it prints accuracy of last 1000 episodes played
#
#
# # In[102]:
#
#
# '''
# import matplotlib.pyplot as plt
#
# plt.plot(current_accuracy, label='Win Rate of '+player)
# plt.plot([1-i for i in current_accuracy], label='Win Rate of '+player1)
#
#
#
# plt.xlabel('TOTAL')
# plt.ylabel('LAST 100')
#
# plt.legend()
#
# plt.show()
# '''
#
#
# # # Saving Model (MLP Object)
#
# # In[103]:
#
#
# import pickle
# with open(f'B00832852_Model.pkl', 'wb') as file:
#     pickle.dump(agent_model, file)
#
# # In[107]:
#
#
# # Testing and accuracy
#
#
# # In[114]:
#
#
# '''
# from backgammon import backgammon
# import numpy as np
# from tqdm import tqdm
#
#
# ls=[]
# episodes=5000
# for i in tqdm(range(0, episodes)):
#     game=backgammon()
#     while game.get_winner() is None:
#         for move in game.get_moves():
#             # Generate a scalar value using the agent's model
#
#             state = game.board.copy()
#             state.extend(move)
#
#             ##TODO: IF the player is white:
#             if game.board[14] == 1:
#                 score = agent_model.return_value(move)
#             # IF the player is holding black checkers
#             else:
#                 score = random.random()
#
#             # Call the store_move function to store the move and scalar value
#             game.score_move(move, score)
#
#         # Check if a winner is determined after each iteration
#         if game.get_winner() is not None:
#             ls.append(game.get_winner())
#             break
#
# '''
#
#
# # In[115]:
#
#
# #print("Total accuracy of testing 5000 episodes for {} win is =".format(player),ls.count(player)/episodes*100)
#
#
# # In[ ]:





