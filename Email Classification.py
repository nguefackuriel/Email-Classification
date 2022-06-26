#!/usr/bin/env python
# coding: utf-8

# <h1>Email Classification using Random Forest, SVM and ANN</h1>

# ## 1) Objectives of this Workshop:
# 
# - Preprocessing an email dataset using Vectorizer
# - Data Exploration using pie chart and world cloud
# - Data Modeling
# - Build the Confusion matrix

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# # 1. **Data Pre-processing**

# #  *Exemple of Vectorizer*

# Preprocessing of text data

# In[ ]:


# example of differents emails

messages = ['I am coming to see you',
              "PLEASE don't let me like that!",
              'Mr. TAMO is a True and good farmer',
              'We will choose one number between one, 2 and 11']


# We have to clean the data using regex, matching patterns in the e-mail messages, and replace them with more organized counterparts. Cleaner data leads to a more efficient model and higher accuracy. Following steps are involved in pre-processing the messages :
# 
# - Lower case every data
# - Remove punctuation
# - Remove the stops words
# 

# In[ ]:


##  initialize our method 

vectorizer = CountVectorizer(stop_words = set(stopwords.words('english')))


# In[ ]:


vectorizer.fit(messages)


# In[ ]:


## the name of our features call agains tokens

vectorizer.get_feature_names()


# In[ ]:


#transform the dataset

messages_transf = vectorizer.transform(messages)
messages_transf


# In[ ]:


#convert it to a dense matrix

messages_transf.toarray()


# In[ ]:


# Generate a Dataframe
pd.DataFrame(messages_transf.toarray(), columns=vectorizer.get_feature_names())


# In[ ]:


# We don't remove the stops words
vectorizer2 = CountVectorizer(stop_words=None)


# In[ ]:


vectorizer2.fit(messages)


# In[ ]:


vectorizer2.get_feature_names()


# In[ ]:


#Total length removal

a=len(vectorizer2.get_feature_names()) # before remove stop words
b=len(vectorizer.get_feature_names()) # after remove stop words
print("Original Length:",a)
print("Cleaned Length:",b)
print("Total Words Removed:",a - b)


# Let us try another method whith TfidfVectorizer method
# 
# First of all, we need to put our message on the table

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


data  = pd.DataFrame(messages)
data.columns =  ['emails']
data


# In[ ]:


vectorizer3 = TfidfVectorizer(stop_words = set(stopwords.words('english')))


# In[ ]:


stop_words=vectorizer3.fit(data)
stop_words


# In[ ]:


features = vectorizer3.fit_transform(data['emails'])


# In[ ]:


features


# In[ ]:


print(features)


# In[ ]:


pd.DataFrame(features.toarray(), columns=vectorizer3.get_feature_names())


# In[ ]:



from google.colab import drive
drive.mount('/content/gdrive')


# ### Now let's take our emails dataset csv file and build our ML code

# # 3. **Data Preprocessing**
# 
# 

# In[ ]:


email = pd.read_csv("/content/gdrive/MyDrive/Email Classification Algorithm/emails.csv")
email.head()


# In this step, we explore data and gain insights such as shape ( form ), structure, type of mail, and percentage of each type.
# 
# First, we check the information of the given dataset and extract information about the dataset

# In[ ]:


stop_word= stopwords.words('english')
for word in stop_word:
  if word in email.columns:
    email.drop(word, axis=1, inplace=True) # delete the stop word on the columns


# In[ ]:


print(email.head(5))


# In[ ]:


# Rename the prediction column into label

email=email.rename(columns={'Prediction' : 'label'})
print(email.head(5))


# ## Investigating the shape of the dataset

# In[ ]:


email.info()


# In[ ]:


print("The shape of the dataset is:", email.shape)


# In[ ]:



print("Count of label: ", email['label'].value_counts())


# For the rest of the work, we consider that label 1 corresponds to Spam mail and lable 0 corresponds to Non spam mail 
# 

# In[ ]:


## Proceeding with Checking Ratio or percentage of Labels i.e. Spam and Non-Spam emails

print("Percentage of Non Spam mail:",round(len(email[email['label']
                                      ==0])/len(email['label']),2)*100,"%")
print("Percentage of Spam mail:",round(len(email[email['label']
                                      ==1])/len(email['label']),2)*100,"%")


# # 4- **Exploratory Data Analysis (EDA)**

# In[ ]:


print("Visualizing ratio non - Spam/Spam:\n")
count = pd.value_counts(email['label'], sort=True) # count the number of 1 and 0
count.plot(kind = 'pie',labels=['non - Spam','Spam'], autopct='%1.0f%%,')
plt.ylabel('')
plt.show


# ## visualization Word Cloud

# Word Cloud is a data visualization technique used for representing text data in which the size of each word indicates its frequency or importance.

# In[ ]:


#Now let's add a string value instead to make our Series clean
colt = list(email.columns)
join_word=" ".join(colt)

# Create and generate a word cloud image:
plt.figure(figsize=(15,15))
wc = WordCloud(background_color="black",  max_words=2000, max_font_size= 300,  width=1600,
               height=800)
wc.generate(join_word)

# Display the generated image:
plt.imshow(wc.recolor( colormap= 'viridis' , random_state=17), interpolation="bilinear")
plt.axis('off')


# ### Let's define our Input and Output columns

# In[ ]:


# Here we split our feature and our prediction to do test and training.

Input = email.iloc[:,1:-1] # extract all the features without  the first and the last column
Output = email['label']   # extract the output
print(Input.shape)
print(Output.shape)


# In[ ]:


Input.head()


# In[ ]:


Output.head()


# ## Build the ML Algorithm

# After splitting the data into test and train using train test split function, we will apply the Random Forest classifier.

# In[ ]:



from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
import seaborn as sns


# In[ ]:


# Splitting data into test and train

x_train, x_test, y_train, y_test = train_test_split(Input,Output, test_size=0.2, random_state=42)


# We are performing a train test split on the dataset. We are providing the test size as 0.2, that means our training sample contains 4138 training set and test sample contains 1034 test set

# # Random Forest classifier

# Random forests is a supervised learning algorithm. It can be used both for classification and regression. here we will use it for classification. A forest is comprised of trees. It is said that the more trees it has, the more robust a forest is. Random forests creates decision trees on randomly selected data samples, gets prediction from each tree and selects the best solution by means of voting

# In[ ]:


# Random Forest classifier with the train test split function.

#Create a Gaussian Classifier
RandomForest =  RandomForestClassifier(n_estimators=100, criterion="gini") # number of trees and to measure the quality of a split.

#Train the model using the training sets 
RandomForest.fit(x_train,y_train)

# prediction on test set
y_predRFC = RandomForest.predict(x_test)

# Model Accuracy, how often is the classifier correct
print(classification_report(y_predRFC, y_test))
print("Accuracy Score of Random Forest Classifier : ", accuracy_score(y_predRFC,y_test))


# In[ ]:


# Confusion matrix
cm = confusion_matrix(y_test, y_predRFC)

# gives the name for each group
group_names = ['True Neg','False Pos','False Neg','True Pos']

# count the number of labels of each group and save
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]

# assign each value to each group
labels = [f"{v1}\n{v2}" for v1, v2, in zip(group_names,group_counts)]

# put our confusion matrix in the matrix with 2 rows and 2 columns
labels = np.asarray(labels).reshape(2,2)

#design our matrix with color
sns.heatmap(cm, annot=labels, fmt='')

plt.title('CONFUSION MATRIX')
plt.xlabel('predicts labels')
plt.ylabel('True labels')


# # Support Vector Machine

# In[ ]:


#Create a SVM Classifier
SVM = SVC(C=1.0, kernel='linear', degree=3 , gamma='auto')

#Train the model using the training sets y_predSVM=rfc.predict(test_x)
SVM.fit(x_train,y_train)

# prediction on test set
y_predSVM = SVM.predict(x_test)

# Model Accuracy, how often is the svm correct
print(classification_report(y_predSVM, y_test))
print("Accuracy Score of SVM : ", accuracy_score(y_predSVM,y_test))


# In[ ]:


# Confusion matrix
cm_svm = confusion_matrix(y_test, y_predSVM)
group_names_svm = ['True Neg','False Pos','False Neg','True Pos']
group_counts_svm = ["{0:0.0f}".format(value) for value in cm.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2, in zip(group_names_svm,group_counts_svm)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm_svm, annot=labels, fmt='')
plt.title('CONFUSION MATRIX')
plt.xlabel('predicts labels')
plt.ylabel('True labels')


# # Artificial Neural Network

# Artificial Neural Networks(ANN) are part of supervised machine learning where we will be having input as well as corresponding output present in our dataset. From the perspective of this blog, we will be developing an ANN for solving the classification class of our problems and compare it to others methods.

# In[ ]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

#Initialising ANN
ANN = Sequential()


#  Once we initialize our ann, we are now going to create layers for the same. Here we are going to create a network 
# that will have 2 hidden layers, 1 input layer, and 1 output layer. So, let’s create our very first hidden layer
# 
# 1. units:- number of neurons that will be present in the respective layer
# 2. activation:- specify which activation function to be used

# In[ ]:


#Adding First Hidden Layer
# we will use “relu”[rectified linear unit] as an activation function for hidden layers.
ANN.add(Dense(units=6,activation="relu")) 

#Adding Second Hidden Layer 
ANN.add(Dense(units=6,activation="relu"))

#Adding Output Layer
ANN.add(Dense(units=1,activation="sigmoid"))

#Compiling ANN
#1. optimizer:- specifies which optimizer to be used in order to perform stochastic gradient descent.
#2. loss:- specifies which loss function should be used.
#3.  metrics:- which performance metrics to be used in order to compute performance

ANN.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

x_train = np.array(x_train) # Put train_x in a matrix

#Fitting ANN
# 1.batch_size: how many observations should be there in the batch. Usually, 
        #the value for this parameter is 32 
# 2. epochs: How many times neural networks will be trained. Here the optimal value that 
      #we have found from our experience is 100.

ANN.fit(x_train, y_train,batch_size=32,epochs = 15)


# In[ ]:


accuracy_score=ANN.evaluate(x_train,y_train)


# In[ ]:


x_test = np.array(x_test)
accuracy_score=ANN.evaluate(x_test,y_test)


# In[ ]:




