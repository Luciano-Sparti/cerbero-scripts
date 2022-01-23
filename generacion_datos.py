# USO:

TrainingImagePath='/home/lucianosp/Documents/Apuntes Facultad/04 - Seminario de Integración Profesional/Sprint 3 - Entrenamiento de Modelo/dataset'

from keras.preprocessing.image import ImageDataGenerator

# Se definen transformaciones de la imagen para alterar el input de las imagenes, para generar un mejor modelo.
train_datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

# No se realizan transformaciones en las imagenes de prueba de modelo.
test_datagen = ImageDataGenerator()

# Generación del set de entrenamiento.
training_set = train_datagen.flow_from_directory(
        TrainingImagePath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')


# Generación del set de prueba.
test_set = test_datagen.flow_from_directory(
        TrainingImagePath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

test_set.class_indices 
