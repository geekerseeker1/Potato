import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from tkinter import *
from tkinter import font

def tf_no_warning():
    """
    Make Tensorflow less verbose
    """
    try:

        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    except ImportError:
        pass
tf_no_warning()
def train_img():
    num_classes = 3
    img_rows, img_cols = 32, 32
    batch_size = 12

    from keras.preprocessing.image import ImageDataGenerator
    train_data_dir = 'data/train/'
    validation_data_dir = 'data/validation/'
    train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')

    validation_datagen = ImageDataGenerator(rescale=1./255)
    #train dataset handling
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    #validation dataset handling

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',input_shape= (img_rows, img_cols, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    print(model.summary())
    # initiate RMSprop optimizer and configure some parameters
    #opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    from keras.optimizers import Adam, SGD
    from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

    checkpoint = ModelCheckpoint("multi_potato_model.h5",monitor="val_loss",mode="min",save_best_only = True,verbose=1)
    earlystop = EarlyStopping(monitor = 'val_loss', #value being monitored for improvement
                          min_delta = 0.1,  #Abs value and is the main change required before we stop
                          patience = 25, #no of epocs we wait before stopping
                          verbose = 1,
                          restore_best_weights = True) #keep the best weigts once stopped
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                              factor = 0.1,
                              patience = 3,
                              verbose = 1,
                              min_delta = 0.001)
    # we put our call backs into a callback list
    callbacks = [earlystop, checkpoint, reduce_lr]
    # We use a very small learning rate
    model.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(lr = 0.001),
              metrics = ['accuracy'])
    nb_train_samples = 9600
    nb_validation_samples = 2400
    epochs = 30
    history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)
    # model.save("multi_potato_model.h5")
    #Displaying our Confusion Matrix
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
    import matplotlib
    from keras.models import load_model
    img_row, img_height, img_depth = 32,32,3
    model = load_model('multi_potato_model.h5')
    class_labels = validation_generator.class_indices
    class_labels = {v: k for k, v in class_labels.items()}
    classes = list(class_labels.values())
    nb_train_samples = 9600
    nb_validation_samples = 2400
    #Confution Matrix and Classification Report
    Y_pred = model.predict_generator(validation_generator, nb_validation_samples // batch_size+1)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(validation_generator.classes, y_pred))
    print('Classification Report')
    target_names = list(class_labels.values())
    print(classification_report(validation_generator.classes, y_pred, target_names=target_names))




def mainScreen():
	root = Tk()
	root.title('Main Page')
	root.geometry('900x800')
	titlefonts = font.Font(family='Helvetica', size=20, weight='bold')
	Label(root,font=titlefonts,text="Potato Leaf Disease Identification System").grid(row=0,column=3,padx=130,sticky=W)

	loginB = Button(root,font=100,text='Start Training', command=train_img)
	loginB.grid(row=40,column=3,columnspan=2,pady=40)
	root.mainloop()

mainScreen()
