from flask import Flask, render_template, request, Markup, redirect
import pickle
import numpy as np
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")
@app.route("/yield",methods=['GET', 'POST'])
def yield1():
    if request.method == 'POST':
        if request.form.get('yi') == 'yie':
            Nitrigen=int(request.form.get("Nitrigen"))
            Phosphorous=int(request.form.get("Phosphorous"))
            Potassium=int(request.form.get("Potassium"))
            Temperature=int(request.form.get("Temperature"))
            Humidity=int(request.form.get("Humidity"))
            PH=int(request.form.get("PH"))
            Rainfall = int(request.form.get("Rainfall"))
            loaded_model = pickle.load(open('knnpickle_file', 'rb'))
            prediction = loaded_model.predict((np.array([[Nitrigen,
                                                          Phosphorous,
                                                          Potassium,
                                                          Temperature,
                                                          Humidity,
                                                          PH,
                                                          Rainfall ]])))
            return render_template("yield.html",value=prediction)

        elif request.form.get('action2') == 'VALUE2':
            pass  # do something else
        else:
            pass  # unknown
    elif request.method == 'GET':
        return render_template('yield.html')

    return render_template("yield.html")
@app.route("/disease",methods=['GET', 'POST'])
def disease1():
    if request.method == 'POST':
        import requests
        import config
        import pickle
        import io
        import torch
        import numpy as np
        import pandas as pd
        from disease import disease_dic
        from torchvision import transforms
        from PIL import Image
        from model import ResNet9
        disease_classes = ['Apple___Apple_scab',
                           'Apple___Black_rot',
                           'Apple___Cedar_apple_rust',
                           'Apple___healthy',
                           'Blueberry___healthy',
                           'Cherry_(including_sour)___Powdery_mildew',
                           'Cherry_(including_sour)___healthy',
                           'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                           'Corn_(maize)___Common_rust_',
                           'Corn_(maize)___Northern_Leaf_Blight',
                           'Corn_(maize)___healthy',
                           'Grape___Black_rot',
                           'Grape___Esca_(Black_Measles)',
                           'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                           'Grape___healthy',
                           'Orange___Haunglongbing_(Citrus_greening)',
                           'Peach___Bacterial_spot',
                           'Peach___healthy',
                           'Pepper,_bell___Bacterial_spot',
                           'Pepper,_bell___healthy',
                           'Potato___Early_blight',
                           'Potato___Late_blight',
                           'Potato___healthy',
                           'Raspberry___healthy',
                           'Soybean___healthy',
                           'Squash___Powdery_mildew',
                           'Strawberry___Leaf_scorch',
                           'Strawberry___healthy',
                           'Tomato___Bacterial_spot',
                           'Tomato___Early_blight',
                           'Tomato___Late_blight',
                           'Tomato___Leaf_Mold',
                           'Tomato___Septoria_leaf_spot',
                           'Tomato___Spider_mites Two-spotted_spider_mite',
                           'Tomato___Target_Spot',
                           'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                           'Tomato___Tomato_mosaic_virus',
                           'Tomato___healthy']

        disease_model_path = 'plant_disease_model.pth'
        disease_model = ResNet9(3, len(disease_classes))
        disease_model.load_state_dict(torch.load(
            disease_model_path, map_location=torch.device('cpu')))
        disease_model.eval()

        def predict_image(img, model=disease_model):
            """
            Transforms image to tensor and predicts disease label
            :params: image
            :return: prediction (string)
            """
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
            ])
            image = Image.open(io.BytesIO(img))
            img_t = transform(image)
            img_u = torch.unsqueeze(img_t, 0)

            # Get predictions from model
            yb = model(img_u)
            # Pick index with highest probability
            _, preds = torch.max(yb, dim=1)
            prediction = disease_classes[preds[0].item()]
            # Retrieve the class label
            return prediction
        file = request.files.get('file')
        img = file.read()
        prediction = predict_image(img)
        prediction = Markup(str(disease_dic[prediction]))
        return render_template('disease-result.html', prediction=prediction)


    elif request.method == 'GET':
        return render_template('disease.html')

    return render_template("disease.html")
@app.route("/weed",methods=['GET', 'POST'])
def weed1():
    if request.method == 'POST':
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        import numpy as np
        from cv2.dnn import readNet
        from torchvision import transforms
        from PIL import Image
        import io
        import os

        net = readNet('yolov3_custom_last.weights', 'yolov3.cfg')
        classes = ['crop', 'weed']
        file = request.files.get('file')
        img = np.array(Image.open(file))
        img = cv2.resize(img, (1280, 720))
        hight, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

        net.setInput(blob)

        output_layers_name = net.getUnconnectedOutLayersNames()

        layerOutputs = net.forward(output_layers_name)

        boxes = []
        confidences = []
        class_ids = []
        for output in layerOutputs:
            for detection in output:
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * hight)
                    w = int(detection[2] * width)
                    h = int(detection[3] * hight)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, .8, .4)
        font = cv2.FONT_HERSHEY_PLAIN
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i],2))

                cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)
                cv2.putText(img, label + " " + confidence, (x + 30, y + 400), font, 10, (0,0,255), 7)
                filename = 'weed_detect.jpg'
                # Using cv2.imwrite() method
                # Saving the image
                path = 'E:\\flask_hackthon\\static'
                cv2.imwrite(os.path.join(path , filename), img)
                # closing all open windows
                cv2.destroyAllWindows()

        return render_template('weed_result.html')

    elif request.method == 'GET':
        return render_template('weed.html')

if __name__ =="__main__":
    app.run(debug=True)