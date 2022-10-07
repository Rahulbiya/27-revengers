import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
im="2.JPG"
def detect():
    global im, result, percentage
    print(f'Image : {im}')
    # resolution
    ht=50
    wd=50
    classNames = ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy" , "Potato___Early_blight" , "Potato___healthy" ,  "Potato___Late_blight" ,
                  "Tomato_Bacterial_spot","Tomato_Early_blight","Tomato_healthy",
                  "Tomato_Late_blight","Tomato_Leaf_Mold","Tomato_Septoria_leaf_spot",
                  "Tomato_Spider_mites_Two_spotted_spider_mite","Tomato__Target_Spot",
                  "Tomato__Tomato_mosaic_virus","Tomato__Tomato_YellowLeaf__Curl_Virus"]
    totClass = len(classNames)
    #print(classNames)
    #print(totClass)
    mdl = r"LeafDisease50x50.h5"
    image = cv2.imread(im)
    orig = image.copy()
    try:
        image = cv2.resize(image, (ht, wd))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
    except Exception as e:
        print("Error Occured : ",e)
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(mdl)
    (zero, one,two, three,four,five,six,seven, eight,nine, ten , eleven, twelve , thirteen , fourteen) = model.predict(image)[0]
    prob = [zero, one,two, three,four,five,six,seven, eight,nine, ten , eleven, twelve , thirteen , fourteen]

    maxProb = max(prob)
    maxIndex = prob.index(maxProb)
    label = classNames[maxIndex]
    proba = maxProb
    result = label
    percentage = float("{0:.2f}".format(proba * 100))
    for i in range(0,totClass):
        print(f'{classNames[i]} : {prob[i]}')
    print(result)
    print(percentage)
detect()
