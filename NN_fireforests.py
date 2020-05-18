# Neural Network forest fires problem

# Artificial Neural Network

## Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('forestfires.csv')
dataset.columns
dataset.describe()
dataset.info()
dataset["size_category"].value_counts()

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('forestfires.csv')
X = dataset.iloc[:, 2:-1].values
y = dataset.iloc[:, [30]].values
print(X)
print(y)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print(X)

# Encoding categorical data
# Label Encoding the "output" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)  # small=1, large =0


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Part 2 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=28, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Part 4 - Making the predictions and evaluating the model
#predicting on training 
y_pred_train = ann.predict(X_train)
np.place(y_pred_train, y_pred_train>0.5, int(1)) 
np.place(y_pred_train, y_pred_train<=0.5,int (0))
y_pred_train = (y_pred_train > 0.5)
print(np.concatenate((y_pred_train.reshape(len(y_pred_train),1), y_train.reshape(len(y_train),1)),1))

y_train=y_train.reshape(len(y_train),1)
y_test=y_test.reshape(len(y_test),1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(y_train, y_pred_train)
print(cm_train)
accuracy_train = np.mean(y_train==y_pred_train)
accuracy_train

# Predicting the Test set results
y_pred_test = ann.predict(X_test)
y_pred_test = (y_pred > 0.5)
print(np.concatenate((y_pred_test.reshape(len(y_pred_test),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred_test)
print(cm_test)
accuracy_test = np.mean(y_test==y_pred_test)
accuracy_test
