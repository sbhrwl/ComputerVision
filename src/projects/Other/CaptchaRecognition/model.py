import cv2
import string
import numpy as np 
%matplotlib inline 
import matplotlib.pyplot as plt
%matplotlib inline 
import os 

import tensorfow as tf
from tensorfow.keras import layers
from tensorfow.keras.models import Model
from tensorfow.keras.models import load_model

print(os.listdir("../input")
sym = string.ascii_lowercase + "0123456789"
num_sym = len(sym)

''' shape of an image '''
img_shape = (50, 200, 1)
''' Number of symbols '''
print(num_sym)
''' Model Creation '''
def Model():
    ''' Input layer '''
    img = layers.Input(shape=img_shape)
    
    ''' Convolutional Layer '''
    out = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(img)
    out = layers.MaxPooling2D(padding='same')(out) 
    
    ''' Convolutional Layer '''
    out = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(out)
    out = layers.MaxPooling2D(padding='same')(out)  
    
    ''' Convolutional Layer '''
    out = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(out)
    out = layers.BatchNormalization()(out)
    out = layers.MaxPooling2D(padding='same')(out)  
    
    ''' flattening output '''
    out = layers.Flatten()(out)
    
    outputs = []
    for _ in range(5):
        ''' Hidden layers '''
        out = layers.Dense(64, activation='relu')(out)
        out = layers.Dropout(0.5)(out)
        
        ''' Classification Layer '''
        out = layers.Dense(num_sym, activation='sigmoid')(out)

        outputs.append(out)
    
    ''' compile the model '''
    model = Model(img, outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
    
    return model
''' Preprocessing '''
def preprocessing():
    samp = len(os.listdir('../input/captcha-version-2-images/samples/samples'))
    X = np.zeros((samp, 50, 200, 1))
    y = np.zeros((5, samp, num_sym))

    for i, p in enumerate(os.listdir('../input/captcha-version-2-images/samples/samples')):
        ''' reading images '''
        img = cv2.imread(os.path.join('../input/captcha-version-2-images/samples/samples', p), cv2.IMREAD_GRAYSCALE)
        p_target = p[:-4]
        if len(p_target) < 6:
            ''' scaling '''
            img = img / 255.0
            ''' reshpaing '''
            img = np.reshape(img, (50, 200, 1))
            
            t = np.zeros((5, num_symb))
            for j, l in enumerate(p_target):
                idx = symbols.find(l)
                t[j, idx] = 1
            X[i] = img
            y[:, i] = t
    
    ''' returning X and y '''
    return X, y

X, y = preprocessing ()

''' train test data '''
X_train, y_train = X[:970], y[:, :970]
X_test, y_test = X[970:], y[:, 970:]
''' calling Model functino '''
model = Model();

''' lets see how model looks like '''
model.summary();
''' training '''
history = model.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], batch_size=32, 
                 epochs=30,verbose=1, validation_split=0.2)
''' predicion funtion to predic an image '''

def predicion(file_path):
    ''' reading an image '''
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    if img is not None:
        ''' scaling '''
        img = img / 255.0
    else:
        print("Not detected");
        
    ''' prediction '''
    pred = np.array(model.predict(img[np.newaxis, :, :, np.newaxis]))
    
    ''' reshaping '''
    pred = np.reshape(pred, (5, 36))
    idx = []
    
    for a in pred:
        idx.append(np.argmax(a))

    captcha = ''
    for l in idx:
        captcha += sym[l]
        
    return captcha
''' evaluating the model '''

sc = model.evaluate(X_test,[y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]],verbose=1)
print('Test Loss and accuracy:', sc)
''' evalutaion '''
model.evaluate(X_test, [y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]])

''' calling prediction function to predict an image '''
print(prediction('../input/captcha-version-2-images/samples/samples/8n5p3.png'))
print(prediction('../input/captcha-version-2-images/samples/samples/f2m8n.png'))
print(prediction('../input/captcha-version-2-images/samples/samples/dce8y.png'))
print(prediction('../input/captcha-version-2-images/samples/samples/3eny7.png'))
print(prediction('../input/captcha-version-2-images/samples/samples/npxb7.png'))
''' plotting an image '''
img = cv2.imread('../input/captcha/capthaimage/capthaimages/a.png',cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap=plt.get_cmap('gray'))
''' prediction '''
print("Predicted Captcha =", prediction('../input/captcha/capthaimage/capthaimages/a.png'))
