#!/usr/bin/env python
# coding: utf-8

# ## Part 2: Spam email classification using Random Forests (50 Points)
# 
# In this part, you must classify the above data set using Random Forests. The
# code must be written in Python and you can use any Python package to solve
# the question. For this part, you must fill up the following table with the best
# classification accuracy achieved across 20 different seeds.
# 
# Shannon I.G. refers to the Shannon Information Gain. Provide an intuition
# for the results observed for the different hyperparameters used. The program
# script for this part must be named 〈NetID〉 hw4 part2.py.

# In[53]:


#Install packages
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.ticker as mtick
#import matplotlib as plt


# In[54]:


#load the dataframe
df = pd.read_csv("spambase.data", names = range(1,59))
#column 58 is the target


# In[55]:


#lets bring those column names, we know this from HW3
df.columns  = ["word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", 
                      "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet", 
                      "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will", 
                      "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free", 
                      "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit", 
                      "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp", 
                      "word_freq_hpl", "word_freq_george", "word_freq_650", "word_freq_lab", "word_freq_labs", 
                      "word_freq_telnet", "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85", 
                      "word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
                      "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project", "word_freq_re", 
                      "word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(", 
                      "char_freq_[", "char_freq_!", "char_freq_$", "char_freq_hash", "capital_run_length_average", 
                      "capital_run_length_longest", "capital_run_length_total", "target"]


# ## Lets break it into parts

# In[56]:


#lets break it
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

Y = df["target"] #just need the target column for Y

#three last variables
X = df.drop(["target"], axis=1) 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1) #test size 0.3 is for the 70-30 split 


# In[57]:


#now we standarize our training and test variables, not necessary for the Y's
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ## Now, lets import the packages, and apply the model accross 20 different seeds
# 
# Please note that the criterion is an input of the tree.DecisionTreeClassifier in the SKlearn model, so we will just create a loop to go through the 20 different seeds using one and the other.
# 
# Moreover, Shannon Information Gain is called Entropy when applying the SKlearn model. The term entropy (in information theory) goes back to Claude E. Shannon, and that's where the name comes from. What Criterion means as an input, is what is the Information Gain formula going to use as a parameter (either Gini or Shannon).
# 
# We are using the same seed for the split and tree, as mentioned in the recitation.

# In[58]:


#packages needed
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

criterions = ['gini', 'entropy']

estimators = [1,3,5,10,15,20,40,70]

#lets create a dictionary to save this data
best_model_accuracy = dict()

for criterion in criterions:
    
    #we create a key for the criterions
    best_model_accuracy[criterion] = {}
    
    #the set of different estimators to fit the model
    for estimator in estimators:
        
        #initialize the estimator key with the criterion corresponding
        model_accs[criterion][estimator] = 0
        
        #now we go over the 20 seeds, and pick the best accuracy of the 20 trials using each estimator
        for seed in range (1,21):
            
            #we do the split and normalization
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = seed) #test size 0.3 is for the 70-30 split 
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            #Now the model
            clf = RandomForestClassifier(n_estimators = estimator, criterion = criterion, random_state = seed)
            clf = clf.fit(X_train, Y_train)
            pred = clf.predict(X_test)
            current_accuracy = accuracy_score(Y_test, pred)
            
            #now we test if the lastest result with a different seed was better
            if current_accuracy > model_accs[criterion][estimator]:
                 best_model_accuracy[criterion][estimator] = current_accuracy
   
    
final_data = pd.DataFrame.from_dict(best_model_accuracy, orient = 'index')


# In[59]:


final_data


# First, lets note that n_estimators stands fot the number of trees in the forest. Then, as the number of trees grows, it does not always mean the performance of the forest is significantly better than previous forests (fewer trees), and doubling the number of trees is sometimes worthless. It is also possible to state there is a threshold beyond which there is no significant gain, unless a huge computational environment is available. And that is sort of the behavior we see on this example: as the number of trees goes higher, the accuracy goes higher as well, but comparing 40 to 70 trees, this accuracy did not increase substantially, and it opens a good conversation into what is the benefit and compuational cost of adding more trees. 
# 
# So the more trees, the better because it has more categories to subdivide the data and improve accuracy, but up to a certain point (in terms of price-cost computational analysis).
# 
# However, it has better accuracy than the trees from part 1 of the HW, with way less nodes involved in the process, which is interesting. Overall, is the best classification model we have produced in the class so far. Moreover, having a word on random_state, in Random Forest Classifier and Regression, random_state controls the randomness of the bootstrapping of the samples used when building trees and the sampling of the features to consider when looking for the best split at each node. So it also plays a descent role in improving the accuracy of the data (making it a little bit more bulletproof).
