import tkinter.messagebox
import cv2
import numpy as np
import time
import cv2
import glob
import pandas as pd
from os.path import join
import os
from PIL import Image
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder
import os
from os.path import join
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import pandas as pd
import time
import cv2
from tkinter import *
import tkinter.messagebox

root=Tk()
root.geometry('1000x400')
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH,expand=1)
root.title('Radiology')
frame.config(background='light blue')
label = Label(frame, text="Radiology Assistant",bg='light blue',font=('Times 35 bold'))
label.pack(side=TOP)

filename = PhotoImage(file=r'C:\Users\Jayesh\Desktop\Project\assist.jpg')
background_label = Label(frame,image=filename)
background_label.pack(side=TOP)
   


def hel():
   help(cv2)

def Contri():
   tkinter.messagebox.showinfo("Contributors","\n1.Amit Kumar\n2. Jayesh Mehta \n3. Yash Wanve \n")


def anotherWin():
   tkinter.messagebox.showinfo("About",'Detect X-ray objects and trigger the Model to generate diagnosis result')
                                    
   

menu = Menu(root)
root.config(menu=menu)

subm1 = Menu(menu)
menu.add_cascade(label="Tools",menu=subm1)
subm1.add_command(label="Open CV Docs",command=hel)

subm2 = Menu(menu)
menu.add_cascade(label="About",menu=subm2)
subm2.add_command(label="Radiology",command=anotherWin)
subm2.add_command(label="Contributors",command=Contri)



def exitt():
   exit()

  
def cam():
    import cv2
    from keras.models import load_model
    import numpy as np
    
    cam = cv2.VideoCapture(0)
    
    cv2.namedWindow("test")
    

    
    while True:
        ret, frame = cam.read()
        #cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(500)
    
        if k%256 == 27:

            print("Escape hit, closing...")
            break
        elif k%256 == 32:

            img_name = ("input_from_cam.png")
            cv2.imwrite(img_name, frame)
            print('image saved')
        
        img = cv2.imread(r'C:\Users\Jayesh\Desktop\Cam\input_from_cam.png')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

        retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255,type=cv2.THRESH_BINARY_INV)   
        image, contours, hierarchy = cv2.findContours(thresh_gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                                           
        mx = (0,0,0,0)    
        mx_area = 0
        for cont in contours:
            x,y,w,h = cv2.boundingRect(cont)
            area = w*h
            if area > mx_area:
                mx = x,y,w,h
                mx_area = area
        x,y,w,h = mx
        
        roi=img[y:y+h,x:x+w]
        cv2.imshow('crop', roi)
        cv2.imwrite('radiology_input.png', roi)
        
        
        cv2.rectangle(img,(x,y),(x+w,y+h),(200,0,0),2)

        
        model = load_model('model_128.h5')
    
    
        model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    
        img = cv2.imread(r'C:\Users\Jayesh\Desktop\Cam\radiology_input.png',cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(128,128))
        img = np.reshape(img,[1,128,128])
        
        img_new = np.expand_dims(img, axis=0)
        img_new.shape
        
        classes = model.predict(img_new, verbose=0, steps=None)
        x = classes
        return x

    cam.release()

    cv2.destroyAllWindows()

   
   
def file():
    
    from keras.models import load_model
    import cv2
    import numpy as np
    
    
    model = load_model('model_128.h5')
    
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    
    img = cv2.imread(r'C:\Users\Jayesh\Desktop\File\ef.jpg',cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(128,128))
    img = np.reshape(img,[1,128,128])
    
    img_new = np.expand_dims(img, axis=0)
    img_new.shape
    classes = model.predict(img_new, verbose=0, steps=None)
    x = classes
    return x

    
def predict_file():
    x = file()
    atelac = (x[0][0]) * 100
    eff = (x[0][1]) * 100
    infl = (x[0][2]) * 100
    nof = (x[0][3]) * 100
    print('using input from file')
    
def predict_cam():
    x = cam()
    atelac = (x[0][0]) * 100
    eff = (x[0][1]) * 100
    infl = (x[0][2]) * 100
    nof = (x[0][3]) * 100
    print('using input from cam')

def result_cam():
    
    print('X-ray analysis complete')         
    print('Please wait for pop-up window')
    import tkinter as tk
    import tkinter.messagebox
    x = cam()
    atelac = (x[0][0]) * 100
    eff = (x[0][1]) * 100
    infl = (x[0][2]) * 100
    nof = (x[0][3]) * 100
    
    prediction_value =("Atelectasis " ,atelac , "\n"
                   "Effusion " ,eff, "\n"
                   "Infiltration ", infl, "\n"
                   "No finding " , nof) 
    master = Tk()
    x = prediction_value 
    master.minsize(width=100, height=100)
    w = Label(master, text=x) 
    w.pack() 

    mainloop()

def result_file():
    
    print('X-ray analysis complete')          
    print('Please wait for pop-up window')
    import tkinter as tk
    import tkinter.messagebox
    x = file()
    atelac = (x[0][0]) * 100
    eff = (x[0][1]) * 100
    infl = (x[0][2]) * 100
    nof = (x[0][3]) * 100

    
    sensor_value =("Atelectasis " ,atelac , "\n"
                   "Effusion " ,eff, "\n"
                   "Infiltration ", infl, "\n"
                   "No finding " , nof) 
    master = Tk()
    x = sensor_value 
    master.minsize(width=100, height=100)
    w = Label(master, text=x) 
    w.pack() 

    mainloop()

def reset():
    import os
    import shutil

    shutil.rmtree(r"C:\Users\Jayesh\Desktop\Cam")
    shutil.rmtree(r"C:\Users\Jayesh\Desktop\File")
    os.mkdir(r"C:\Users\Jayesh\Desktop\Cam")
    os.mkdir(r"C:\Users\Jayesh\Desktop\File")




        
but1=Button(frame,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=cam,text='Input using Cam',font=('helvetica 15 bold'))
but1.place(x=50,y=50)

but1=Button(frame,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=file,text='Input using File',font=('helvetica 15 bold'))
but1.place(x=500,y=50)

but1=Button(frame,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=predict_file,text='Predict',font=('helvetica 15 bold'))
but1.place(x=500,y=150)

but1=Button(frame,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=predict_cam,text='Predict',font=('helvetica 15 bold'))
but1.place(x=50,y=150)

but1=Button(frame,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=result_cam,text='Result',font=('helvetica 15 bold'))
but1.place(x=50,y=250)

but1=Button(frame,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=result_file,text='Result',font=('helvetica 15 bold'))
but1.place(x=500,y=250)

but1=Button(frame,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=reset,text='Reset',font=('helvetica 15 bold'))
but1.place(x=275,y=350)

root.mainloop()