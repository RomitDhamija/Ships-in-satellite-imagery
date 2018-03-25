import numpy as np 
import json
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import keras.callbacks

from sklearn.model_selection import train_test_split


def conv_model():
    
    input_shape= (80,80,3)
    num_classes= 2

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model

def see_ships(X_new):
	ship= X_new[1]
	noship= X_new[800]
	plt.subplot(1,2,1)
	plt.title('ship')
	plt.imshow(ship)

	plt.subplot(1,2,2)
	plt.imshow(noship)
	plt.title("no ship")

	plt.show()

def acc_loss(history):
    plt.subplot(1,2,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['train','test'], loc='best')
    plt.title('Accuracy')

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train','test'], loc= 'best')
    plt.title('loss')

    plt.show()



if __name__ == '__main__':

	with open('./data/shipsnet.json') as file_:
	    text= file_.read()
	    data= json.loads(text)
	file_.close()
	df= pd.DataFrame(data)
	# df.head()

	X= np.array(data['data']).astype('float32')
	y= np.array(data['labels']).astype('float32')

	#Preparing Data for model input
	X_new= np.reshape(X, (-1, 3,80,80)).transpose([0,2,3,1])
	X_new= X_new/255.0
	y_new= to_categorical(y, num_classes=2)

	x_train, x_test, y_train, y_test= train_test_split(X_new, y_new, test_size= 0.3, random_state= 23)

	model= conv_model()

	batch_size= 128
	epochs= 15
	datagen= ImageDataGenerator(featurewise_center=False,
	                            samplewise_center=False,
	                            featurewise_std_normalization=False,
	                            samplewise_std_normalization=False,
	                            zca_whitening=False,
	                            zca_epsilon=1e-6,
	                            rotation_range=10,
	                            width_shift_range=0.2,
	                            height_shift_range=0.2,
	                            shear_range=0.,
	                            zoom_range=0.,
	                            channel_shift_range=0.,
	                            fill_mode='nearest',
	                            cval=0.,
	                            horizontal_flip=True,
	                            vertical_flip=False)

	history= model.fit_generator(datagen.flow(x_train, y_train, batch_size= batch_size), 
		                         epochs= epochs, validation_data=[x_test, y_test],
		                         steps_per_epoch= len(x_train)/batch_size)

	score= model.evaluate(x_test, y_test, verbose= 1)
	# print(score)
	acc_loss(history)

	model_json= model.to_json()
	#serialize model to json
	with open('./results/ships_model.json','w') as model_file:
    model_file.write(model_json)

	#serialize weights to hdf5
	model.save_weights('./results/ships_model_weights.h5')

	print('files loaded  and saved...')

	exit(0)