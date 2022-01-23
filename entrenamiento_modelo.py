from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
 
#CNN en modo secuencial
classifier= Sequential()
 
# CNN 1ra capa: Convoluciones
#kernel de 5x5 pixeles, de a 1 paso, con imagenes de 64x64 pixeles x 3 (RGB)
classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64,64,3), activation='relu'))
 
classifier.add(MaxPool2D(pool_size=(2,2)))

# CNN 2da capa, siendo el input el resultado de la primera
classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
 
classifier.add(MaxPool2D(pool_size=(2,2)))
 
# Removemos "spikes": aplanamos curvas para generalizar mejor
classifier.add(Flatten())
 
classifier.add(Dense(64, activation='relu'))
classifier.add(Dense(OutputNeurons, activation='softmax'))
 
# Compilación de la red neuronal
#classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])
 
# Generador
classifier.fit_generator(
    training_set,
    steps_per_epoch=30,
    epochs=10,
    validation_data=test_set,
    validation_steps=10)
