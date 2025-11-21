import numpy as np
import os
import cv2
import matplotlib
from os import listdir
from os.path import isfile, join
import re
from tkinter import *
from tkinter import font
from tkinter import messagebox, filedialog, ttk
import shutil
from PIL import Image, ImageTk, ImageOps
import tkinter as tk
import tensorflow as tf

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

def predict_disease():
    from keras.models import load_model
    from keras.preprocessing import image
    from keras.preprocessing.image import ImageDataGenerator
    model = load_model('multi_potato_model.h5')
    validation_data_dir = 'data/validation/'
    validation_datagen = ImageDataGenerator(rescale=1./255)
    img_rows, img_cols = 32, 32
    batch_size = 12
    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    class_labels = validation_generator.class_indices
    class_labels = {v: k for k, v in class_labels.items()}
    classes = list(class_labels.values())

    def draw_test(name, pred, im, true_label):
        BLACK = [0,0,0]
        print (pred)
        im=cv2.resize(im,(256,256))
        expanded_image = cv2.copyMakeBorder(im, 360, 0, 0, 900 ,cv2.BORDER_CONSTANT,value=BLACK)
        cv2.putText(expanded_image, "predicted - "+ pred, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
        early="Potato___Early_blight"
        late="Potato___Late_blight"
        healthy="Potato___healthy"
        if (pred==early):
            cv2.putText(expanded_image, "Caused By - Alternaria solani", (20, 100) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)
            cv2.putText(expanded_image, "Treatment - Bonide Garden Dust pouder", (20, 140) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,128,128), 2)
            cv2.putText(expanded_image, "Mix 1 pound to 5 gallons of water, Spray to cover all parts of plant.", (20, 170) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,128,128), 2)
            cv2.putText(expanded_image, "Repeat at 7 to 10 day intervals and after rains.", (20, 200) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,128,128), 2)

        if (pred==late):
            cv2.putText(expanded_image, "Caused By - Phytophthora infestans", (20, 100) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)
            cv2.putText(expanded_image, "Treatment - copper based fungicide", (20, 140) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,128,128), 2)
            cv2.putText(expanded_image, "Apply a copper based fungicide (60 ml with a 16 Litre of water)", (20, 170) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,128,128), 2)
            cv2.putText(expanded_image, "Repeat at 7 to 10 day intervals and after rains.", (20, 200) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,128,128), 2)
        
        if (pred==healthy):
            cv2.putText(expanded_image, "Potato leaf is Healthy.", (20, 100) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)
        #cv2.putText(expanded_image, "true - "+ true_label, (20, 120) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)
        cv2.imshow(name, expanded_image)
  

    # dimensions of our images
    img_width, img_height = 32, 32

    files = []
    predictions = []
    true_labels = []
    # predicting images
    path = 'test/' 

    for img_name in os.listdir(path):
        true_labels.append(img_name)
        final_path="test/"+img_name
        img=image.load_img(final_path, target_size = (img_width, img_height))
        x = image.img_to_array(img)
        x = x * 1./255
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict_classes(images, batch_size = 10)
        predictions.append(classes)
        print(predictions)
    image = cv2.imread((final_path))
    draw_test("Prediction", class_labels[predictions[0][0]], image, true_labels[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
def upload_image():
    dest='test'
    img_names=[]
    for the_file in os.listdir(dest):
        file_path = os.path.join(dest, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
        
    filepath=filedialog.askopenfilename()
    shutil.copy(filepath,dest)

    #display image
    for img_name in os.listdir(dest):
        img_names.append(img_name)
  
    load = Image.open("test/"+img_names[0])
    #load=cv2.resize(load,(256,256))
    size = (256, 256)
    load = ImageOps.fit(load, size, Image.ANTIALIAS)
    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render, height="250", width="500")
    img.image = render
    img.place(x=0, y=0)
    img.grid(column=2, row=20, padx=10, pady = 10)
    #title.destroy()
    button2 = tk.Button(font=100,text="Analyse Image", command=predict_disease)
    button2.grid(column=3, row=30, padx=10, pady = 10)
         


def mainScreen():
	root = Tk()
	root.title('Main Page')
	root.geometry('900x800')
	titlefonts = font.Font(family='Helvetica', size=20, weight='bold')
	Label(root,font=titlefonts,text="Potato Leaf Disease Identification System").grid(row=0,column=3,padx=130,sticky=W)

	loginB = Button(root,font=100,text='Upload Image to test', command=upload_image)
	loginB.grid(row=40,column=3,columnspan=2,pady=40)
	root.mainloop()

mainScreen()
