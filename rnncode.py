from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tkinter import *
from tkinter import ttk,StringVar,IntVar
from tkinter import filedialog
from PIL import ImageTk, Image
from osgeo import gdal
from tkinter import messagebox
import tkinter.messagebox
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import array
import sys
import os
import string
import time
import matplotlib
import io
from keras.layers import *
from keras import Sequential
from keras.utils import np_utils
from PIL import Image
from contextlib import redirect_stdout

global loc
global fileweightsave
layeradd = []
dataadd = []

def extractdata(fileweightsave):
    layeradd = []
    dataadd = []
    location=fileweightsave.split(".hdf5")
    print(location)
    file = open(location[0]+".txt",'r')
    lines = file.readlines()
    lines = [x.strip() for x in lines]
    t=lines[0].split(",")
    thresh=float(t[2])
    act=str(t[1])
    x=int(t[0])
    y=len(lines)
    for i in range(1,x):
        dt=lines[i].split(",")
        dataadd.append(dt[1])
    for i in range(x,y):
        print(lines[i])
        layeradd.append(lines[i])
    return x,dataadd,layeradd,act,thresh

def extract():
    with open ('saved_val.txt', 'rt') as in_file: # Open file txt for reading of text data.
        for line in in_file:
            tokens=line.split("//null//")
    in_file.close()
    st=tokens[0].split(":")
    train_cycle=int(st[1])
    st=tokens[1].split(":")
    cls=st[1]
    c_c=int(cls)+1
    st=tokens[2].split(":")
    threshold=float(st[1])
    st=tokens[3].split(":")
    activate=str(st[1])
    st=tokens[4].split(":")
    pool=str(st[1])
    st=tokens[6].split(";")
    imgpath=st[1]
    st=tokens[8].split(":")
    row=int(st[1])
    st=tokens[9].split(":")
    col=int(st[1])
    st=tokens[10].split(":")
    bands=int(st[1])
    st=tokens[11].split(":")
    dtypevalue=int(st[1])
    if (dtypevalue == 8):
        dtype=np.uint8
    elif(dtypevalue ==16):
        dtype=np.uint16
    poolc="MaxPooling"
    if(pool==poolc):
        value=1
    else:
        value=2

    return train_cycle,c_c,threshold,activate,pool,imgpath,row,col,bands,value,dtype,dtypevalue

def putimage(result,window3=10):
    window3=tkinter.Toplevel()
    frame = Frame(window3, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    img = ImageTk.PhotoImage(result)

    label = Label(frame, bd=0,image=img)
    label.grid(row=0, column=0, sticky=N+S+E+W)

    label.image=img


    frame.pack(fill=BOTH,expand=1)

def train(datapath,layer):
    train_cycle,c_c,threshold,activate,pool,imgpath,row,col,bands,value,dtype,dtypevalue=extract()
    values = []
    c_l={}
    path=datapath
    c=0
    for add in path:
        c=int(c)+1
        print("{} class {} ".format(add,c))
        c_l[add]=c
    clicks={}

    for address in path:
        with open(address, "rb") as f:
            k = len(f.read())
            clicks[address] = (k // 2 // bands) if (k // 2 // bands) < 400 else (k // 2 // bands) // 4
            print('{} ==> {}'.format(address, clicks[address]))

    for address in path:
        with open(address, "rb") as f:
            b = array.array("H")
            b.fromfile(f, clicks[address]*bands)
            if sys.byteorder == "little":
                b.byteswap()
            for v in b:
                values.append(v)

    ll = (len(values))
    rex = ll // bands
    print(ll, rex)

    f_in = np.zeros([ll], dtype)
    x = 0
    for i in range(ll):
        f_in[x] = values[i]
        x += 1

    y_train = np.zeros([rex], dtype)

    mark = 0
    for add in path:
        for i in range(clicks[add]):
            y_train[mark+i] = c_l[add]
        mark = mark + clicks[add]


    x_train = f_in.reshape(rex, bands)

    seed = 6
    np.random.seed(seed)

    x_train = x_train / (2**(dtypevalue-1))
    num_pixels = bands

    for v in y_train:
        print(v, end=" ")

    y_train = np_utils.to_categorical(y_train)
    n_classes = c_c
    print(x_train)
    print(20*'#')
    print(y_train)

    print(x_train.shape)
    print(y_train.shape)

    X = x_train.reshape(x_train.shape[0], bands, 1)

    n_units=128
    n_classes=c_c
    batch_size=50
    j=3
    t=int(len(layer))-1
    model = Sequential()
    for i in range(0,len(layer)):
        if(layer[i]=="Convolution"):
            if(i==0):
                model.add(Conv1D(2 ** j, 2, activation=activate, padding='same', input_shape=[bands, 1]))
            else:
                model.add(Conv1D(2 ** j, 2, activation="relu", padding='same'))
            j=j+1
        elif(layer[i]=="MaxPooling"):
            model.add(MaxPooling1D(2))
        elif(layer[i]=="AveragePooling"):
            model.add(AveragePooling1D(2))
        elif(layer[i]=="LSTM"):
            if(i==0):
                model.add(LSTM(2**j,return_sequences=False, input_shape=(bands ,1)))
            elif(i==t):
                model.add(LSTM(2**(j-1)))
            else:
                model.add(LSTM(2**j, return_sequences=True))
            j=j+1

    model.add(Dense(n_classes, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    #model.fit(X, y_train, batch_size=10, epochs=10)
    model.fit(X, y_train, batch_size=50, epochs=train_cycle)

    return model,c_c,activate,threshold

def saveweight(c_c,datapath,layer,model_new,activate,threshold):
    MsgBox = messagebox.askquestion ('Select an option','Do you want to save the weights',icon = 'question')
    if MsgBox=="yes":
        fileweightsave=filedialog.asksaveasfilename(initialdir = "/",title = "Select file",defaultextension=".hdf5")
        print(fileweightsave)
        loc=fileweightsave.split(".hdf5")
        file = open(loc[0]+'.txt','w')
        file.write(str(c_c)+","+str(activate)+","+str(threshold)+"\n")
        i=1
        for i in range(0,len(datapath)):
            sdata=datapath[i].split('/')
            file.write(str(i+1)+","+str(sdata[len(sdata)-1])+"\n")
            i=i+1
        for i in range(0,len(layer)):
            file.write(layer[i]+"\n")
        model_new.save_weights(fileweightsave,overwrite=True)

def ReadBilFile(bil,bands,pixels,dtype):
    extract_band=1
    image=np.zeros([pixels,bands], dtype)
    gdal.GetDriverByName('EHdr').Register()
    img = gdal.Open(bil)
    while bands>=extract_band:
        bandx = img.GetRasterBand(extract_band)
        datax = bandx.ReadAsArray()
        temp=datax
        store=temp.reshape(pixels)
        for i in range(pixels):
            image[i][extract_band-1]=store[i]
        extract_band=extract_band+1
    return image

def test(model_new,datapath):
    train_cycle,c_c,threshold,activate,pool,imgpath,row,col,bands,value,dtype,dtypevalue=extract()
    pixels = row * col
    y_test = np.zeros([row * col], dtype)
    x_test = ReadBilFile(imgpath, bands, pixels,dtype)
    x_test = x_test.reshape(row*col, bands, 1)

    x_test = x_test / (2**(dtypevalue-1))
    y_test = np_utils.to_categorical(y_test)

    datapathname=[]
    for i in range(0,len(datapath)):
        sdata=datapath[i].split('/')
        print(sdata)
        datapathname.append(sdata[len(sdata)-1])

    y_test_new = model_new.predict(x_test, batch_size=50)
    maxlim=threshold
    for rno in range(1,pixels):
         for cno in range(1,c_c-1):
            if(y_test_new[rno][cno]<maxlim):
                y_test_new[rno][cno]=0;

    print( 'printing new values of ytestnew' )
    print(y_test_new)
    print(" new values printed")
    print(20*'%')
    print(y_test_new.shape)
    print(20*'%')
    y_test1 = np.argmax(y_test_new, axis=1)
    print(30*'*')
    print("this is predicted output")
    imageresult=[]
    imgname=[]

    k=y_test1.reshape(row,col)
    plt.imshow(k)
    plt.show()
    result = Image.fromarray((k * (2**(dtypevalue-1))//c_c).astype(dtype))
    imageresult.append(result)
    name="hard"
    imgname.append(name)
    putimage(result)

    for i in range(1, c_c):
        img = y_test_new[:,i].reshape(row, col)
        plt.imshow(img*(2**(dtypevalue-1)))
        plt.colorbar()
        plt.show()
        result = Image.fromarray(((img * (2**(dtypevalue-1)))).astype(dtype))
        imageresult.append(result)
        imgname.append(datapathname[i-1])
        putimage(result,i)

    MsgBox = messagebox.askquestion ('Select an option','Do you want to save the images',icon = 'question')
    if MsgBox=="yes":
        loc=filedialog.askdirectory()
        for i in range (0,len(imageresult)):
            imageresult[i].save(loc+'/'+imgname[i]+'_classified.tiff')

def testtrain(layer,weightfile,datapath,row,col,bands,c_c,imgpath,dtype,dvalue,activate,threshold):
    print(layer)
    print(weightfile)
    print(datapath)
    j=3
    t=int(len(layer))-1
    model2 = Sequential()
    for i in range(0,len(layer)):
        if(layer[i]=="Convolution"):
            if(i==0):
                model2.add(Conv1D(2 ** j, 2, activation=activate, padding='same', input_shape=[bands, 1]))
            else:
                model2.add(Conv1D(2 ** j, 2, activation=activate, padding='same'))
            j=j+1
        elif(layer[i]=="MaxPooling"):
            model2.add(MaxPooling1D(2))
        elif(layer[i]=="AveragePooling"):
            model2.add(AveragePooling1D(2))
        elif(layer[i]=="LSTM"):
            if(i==0):
                model2.add(LSTM(2**j,return_sequences=False, input_shape=(bands ,1)))
            elif(i==t):
                model2.add(LSTM(2**(j-1)))
            else:
                model2.add(LSTM(2**j, return_sequences=True))
            j=j+1

    model2.add(Dense(c_c, activation='sigmoid'))
    model2.load_weights(weightfile)

    datapathname=[]
    for i in range(0,len(datapath)):
        sdata=datapath[i].split('/')
        print(sdata)
        datapathname.append(sdata[len(sdata)-1])
    pixels = row * col
    y_test = np.zeros([row * col], dtype)
    x_test = ReadBilFile(imgpath, bands, pixels,dtype)
    x_test = x_test.reshape(row*col, bands, 1)

    x_test = x_test / (2**(dvalue-1))
    y_test = np_utils.to_categorical(y_test)

    y_test = model2.predict(x_test, batch_size=50)
    maxlim=threshold
    for rno in range(1,pixels):
         for cno in range(1,c_c-1):
            if(y_test[rno][cno]<maxlim):
                y_test[rno][cno]=0;

    print( 'printing new values of ytestnew' )
    print(y_test)
    print(" new values printed")
    print(20*'%')
    print(y_test.shape)
    print(20*'%')
    y_test1 = np.argmax(y_test, axis=1)
    print(30*'*')
    print("this is predicted output")
    imageresult=[]
    imgname=[]

    k=y_test1.reshape(row,col)
    plt.imshow(k)
    plt.show()
    result = Image.fromarray((k * (2**(dvalue-1))//c_c).astype(dtype))
    imageresult.append(result)
    name="hard"
    imgname.append(name)
    putimage(result)

    for i in range(1, c_c):
        img = y_test[:,i].reshape(row, col)
        plt.imshow(img*(2**(dvalue-1)))
        plt.colorbar()
        plt.show()
        result = Image.fromarray(((img * (2**(dvalue-1)))).astype(dtype))
        imageresult.append(result)
        imgname.append(datapathname[i-1])
        putimage(result,i)

    MsgBox = messagebox.askquestion ('Select an option','Do you want to save the images?',icon = 'question')
    if MsgBox=="yes":
        loc=filedialog.askdirectory()
        for i in range (0,len(imageresult)):
            imageresult[i].save(loc+'/'+imgname[i]+'_classified.tiff')

def typemain(file):
    if file.endswith('.hdr'):
        f= open(file,"r")
    else:
        return None
    i=1
    for line in f.readlines():
        line = str(line.lower())
        line = line.strip().lower()
        if str(line) == "datatype: u16" or (str(line) == "datatype: s16"):
            D_type=np.uint16
            datavalue=16
        if str(line) == "datatype: u8":
            D_type=np.uint8
            datavalue=8
        i=i+1
    return D_type,datavalue

def headerreader(headpath):
    dtype,dvalue=typemain(headpath)
    i=1
    r=-1
    c=-1
    b=-1
    pat1 = re.compile(r"\brows\b", re.IGNORECASE) # but matches because pattern ignores case
    pat2 = re.compile(r"\bcols\b", re.IGNORECASE) # but matches because pattern ignores case
    pat3 = re.compile(r"\bbands\b", re.IGNORECASE)# but matches because pattern ignores case
    lines = [] #Declare an empty list named "lines"
    with open (headpath, 'rt') as in_file: # Open file txt for reading of text data.
        for line in in_file:
            if pat1.search(line) != None:
                lines.append(line)
                i=i+1
                r=line.split(" ",1)[1]
                print("this is the number of rows extracted:")
                print(r)

            elif pat2.search(line) != None:
                lines.append(line)
                i=i+1
                c=line.split(" ",1)[1]
                print("this is the number of columns extracted:")
                print(c)

            elif pat3.search(line) != None:
                lines.append(line)
                i=i+1
                b=line.split(" ",1)[1]
                print("this is the number of bands extracted:")
                print(b)
            else:
                i=i+1

    if( r==-1 or b==-1 or c==-1):
        print("ERROR: header information insuffficient")

    in_file.close()
    rr=int(r)
    cc=int(c)
    bb=int(b)
    return rr,cc,bb,dtype,dvalue

def trainmodel(datapath,layer):
    model_new,c_c,activate,threshold=train(datapath,layer)
    messagebox.showinfo("Message", "Model Trained Successfully!")
    saveweight(c_c,datapath,layer,model_new,activate,threshold)


def testtrainmodel(datapath,layer):
    model_new,c_c,activate,threshold=train(datapath,layer)
    test(model_new,datapath)
    saveweight(c_c,datapath,layer,model_new,activate,threshold)


def modeltest(weightfile,headerfile,imgpath):
    row,col,band,dtype,dvalue=headerreader(headerfile)
    c_c,datapath,layer,activate,threshold=extractdata(weightfile)
    testtrain(layer,weightfile,datapath,row,col,band,c_c,imgpath,dtype,dvalue,activate,threshold)
