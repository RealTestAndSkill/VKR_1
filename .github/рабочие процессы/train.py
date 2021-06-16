from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
import zagr_izo

img_width = 150
img_height = 150

epoh = 20
batch_size = 20
number_train = 20
number_prover = 20
number_test = 20

# Создание сети
model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(150,150,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Компиляция сети
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Генератор изображений
data_gen = ImageDataGenerator(rescale=1. / 255)
train_gen = data_gen.flow_from_directory(zagr_izo.train, target_size=(img_width,img_height), batch_size=batch_size, class_mode='binary')
test_gen = data_gen.flow_from_directory(zagr_izo.test, target_size=(img_width,img_height), batch_size=batch_size, class_mode='binary')
prover_gen = data_gen.flow_from_directory(zagr_izo.prover, target_size=(img_width,img_height), batch_size=batch_size, class_mode='binary')

model.fit_generator(train_gen, steps_per_epoch=number_train // batch_size, epochs=epoh, validation_data=prover_gen, validation_steps=number_prover // batch_size)
model.save('MandB.h5')


