
from flask import Flask, render_template, request, flash, redirect
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')



@app.route("/malariapredict", methods = ['POST', 'GET'])
def malariapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,3))
                img = img.astype(np.float64)
                model = load_model("models/malaria.h5")
                pred = np.argmax(model.predict(img)[0])
        except:
            message = "  الرجاء تحميل صوره الخلية فقط"
            return render_template('malaria.html', message = message)
    return render_template('malaria_predict.html', pred = pred)

@app.route("/pneumoniapredict", methods=['POST', 'GET'])
def pneumoniapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
               
                img = Image.open(request.files['image'])
                img_rgb = img.convert('RGB')
                img_np = np.array(img_rgb)
                
    
                mean_red = np.mean(img_np[:, :, 0])
                mean_green = np.mean(img_np[:, :, 1])
                mean_blue = np.mean(img_np[:, :, 2])
                
            
                if abs(mean_red - mean_green) > 20 or abs(mean_green - mean_blue) > 20 or abs(mean_red - mean_blue) > 20:
                    message = "رجاء قم بتحميل صورة أشعة سينية مناسبة ."
                    return render_template('pneumonia.html', message=message)
                
               
                img = img.convert('L')  
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,1))
                img = img / 255.0


            
                model = load_model("models/pneumonia.h5")
                pred = np.argmax(model.predict(img)[0])

        except:
            message = "رجاء قم بختيار صورة"
            return render_template('pneumonia.html', message=message)
    
    return render_template('pneumonia_predict.html', pred=pred)

if __name__ == '__main__':
	app.run(debug = True)





