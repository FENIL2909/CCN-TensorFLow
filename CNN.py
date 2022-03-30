
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint


#################################################################
# Insert TensorFlow code here to complete the tutorial in part 1.
################################################################
# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Define the text labels
fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2 
                        "Dress",        # index 3 
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6 
                        "Sneaker",      # index 7 
                        "Bag",          # index 8 
                        "Ankle boot"]   # index 9


#Normalize the data dimensions so that they are of approximately the same scale.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Further break training data into train / validation sets (# put 5000 into validation set and keep remaining 55,000 for train)
(x_train, x_valid) = x_train[5000:], x_train[:5000] 
(y_train, y_valid) = y_train[5000:], y_train[:5000]

# Reshape input data from (28, 28) to (28, 28, 1)
w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))


model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

# Take a look at the model summary
model.summary()

checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only=True)
model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=10,
         validation_data=(x_valid, y_valid),
         callbacks=[checkpointer])

# Load the weights with the best validation accuracy
model.load_weights('model.weights.best.hdf5')

# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1]*100,"%")

y_hat = model.predict(x_test)

# ################################################################
# # Insert TensorFlow code here to *train* the CNN for part 2.
# ################################################################

model1 = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
model1.add(tf.keras.layers.Conv2D(filters=64, strides=1, kernel_size=3, padding='valid', activation= None, input_shape=(28,28,1))) 
model1.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2), padding='valid'))
model1.add(tf.keras.layers.Activation('relu'))
model1.add(tf.keras.layers.Flatten())
model1.add(tf.keras.layers.Dense(1024, activation='relu'))
model1.add(tf.keras.layers.Dense(10, activation='softmax'))

model1.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

# Take a look at the model summary
model1.summary()

checkpointer = ModelCheckpoint(filepath='model1.weights.best.hdf5', verbose = 1, save_best_only=True)
model1.fit(x_train,
         y_train,
         batch_size=64,
         epochs=10,
         validation_data=(x_valid, y_valid),
         callbacks=[checkpointer])

# Load the weights with the best validation accuracy
model1.load_weights('model1.weights.best.hdf5')

# Evaluate the model on test set
score = model1.evaluate(x_test, y_test, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1]*100,"%")

#################################################################
# Write a method to extract the weights from the trained
# TensorFlow model. In particular, be *careful* of the fact that
# TensorFlow packs the convolution kernels as KxKx1xF, where
# K is the width of the filter and F is the number of filters.
#################################################################

def convertWeights (model1):
    # Extract W1, b1, W2, b2, W3, b3 from model.
    # Convert Convolution Weights to fully connected weights
    w1=model1.layers[0].get_weights()[0]    # Extracting convolution filters for first layer
    W1= np.zeros((26*26*64,28*28))   #Initilaizing all the weights of fully connected layer as zero
    
    B1=model1.layers[0].get_weights()[1]    #Extracting the Bias for first layer
    b1=np.zeros(26*26*64)  #Initializing the bias as zero for layer 1 of fully the connected layer
    vboost=0     #This will help add the extra zeros to add in the row when filter moves in vertical direction
    hboost=0     #This will help add the extra zeros to add in the row when filter moves in horizontal direction
    filter=-1  #This determines the filter number
    
    for i in range(0,26*26*64):    #Looping through the rows of fully connected weights
        #Update the boost when filter moves vertically down       
        if(int(i%26)==0):   
            vboost+=2

        #Update the vboost to zero and increment filter when new filter is encountered
        if(i%(26*26)==0):   
            vboost=0
            filter+=1 

        #Adding the same bias term for each filter
        b1[i]= B1[filter]

        #Updating the horizontal boost after end of each row
        hboost=(i%(26*26))+vboost 

        #Intitalizing the row of the filter as -1
        row=-1  

        for j in range(0,28*28):  #Looping through the columns of the fully connected weights
            col=(j-hboost)%28   #column of the filter
            if (col==0 and (j-hboost)>-1):  #Condition to increase the row index of filter
                row+=1
            if( ((j-hboost)>-1) and (col<3) and (row<3) ):  #Condition to add the filter weight into the fully connected weight
                W1[i][j] = w1[row,col,0,filter]

    #Performing transpose to get the correct shape of the weights
    W1=W1.T
    
    #Extracting other weights and biases from the model
    W2=model1.layers[4].weights[0].numpy()
    b2=model1.layers[4].weights[1].numpy()
    W3=model1.layers[5].weights[0].numpy()
    b3=model1.layers[5].weights[1].numpy()

    return W1, b1, W2, b2, W3, b3

#################################################################
# Below here, use numpy code ONLY (i.e., no TensorFlow) to use the
# extracted weights to replicate the output of the TensorFlow model.
#################################################################

# Implement a fully-connected layer.
def fullyConnected (W, b, x):
    Z=np.dot(W.T,x)+b
    return Z

# Implement a max-pooling layer
def maxPool (x, poolingWidth):
    x=np.reshape(x,(26*64,26))   #Reshaped the flattened vector for easy access of element
    z=[]    #Initializing an empty list to store the maxpooled neurons

    #Looping over the reshaped vector to perform max pooling
    for i in range(0,26*64-poolingWidth+1,poolingWidth):
        for j in range(0,26-poolingWidth+1,poolingWidth):
            #Adding the Pooling Elements to a List
            pooling_elements=[x[i,j],x[i,j+1],x[i+1,j],x[i+1,j+1]]

            #Append the max element into the maxpooled neurons
            z.append(max(pooling_elements))

    return z

# Implement a softmax function.
def softmax (x):
    return np.exp(x)/np.sum(np.exp(x))

# Implement a ReLU activation function
def relu (x):
    return np.maximum(0,x)

# Load weights from TensorFlow-trained model.
W1, b1, W2, b2, W3, b3 = convertWeights(model1)

#Selecting Input Image
x=x_train[0:1,:,:,:]
x=x.flatten()   #Flattening the image for Fully connected Model

#MODEL
# 1. Peform Convolution using the Fully Connected Weights
Z1=fullyConnected(W1,b1,x)

# 2. Perform Maxpooling operation
Z2= maxPool(Z1,2)

# 3. Perform ReLu Activation
h3= relu(Z2)

# 4. Rearranging the flattened vector to match the flanttened vector generated from tensor flow model
#(This is done to avoid the mismatch of the weights we got from the tensor flow model)
h4=[]   #Iniitalizing an empty list to store the new flattened vector
#Traversing through the neurons
for i in range(0,13*13):
    #Traversing the same position in the different feature maps
    for j in range(i,13*13*64,13*13): 
        h4.append(h3[j])    

# 5. Fully Connected layer to map into 1024 Dimensional Vector and applying ReLu activation
Z5=fullyConnected(W2,b2,h4)
h5=relu(Z5)

# 6. Fully Connected layer to map into 10 Dimensional Vector and applying Softmax activation
Z6=fullyConnected(W3,b3,h5)
Y6=softmax(Z6)

# Predicting the softmax output for Tensorflow Model
yhat1 = model1.predict(x_train[0:1,:,:,:])[0]  # Save model's output

#Print the obtained softmax output for Tensorflow Model and Fully Connected Model for same input image
print("\n--------------------------------------------------")
print("Output of Tensorflow Model")
print("--------------------------------------------------")
print(yhat1)
print("\n--------------------------------------------------")
print("Output of Fully Connected Model")
print("--------------------------------------------------")
print(Y6)
print("\n")