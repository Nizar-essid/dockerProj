import librosa
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn import svm
import matplotlib.pyplot as plt
import pickle
from scipy.stats import norm
from scipy import stats
from os import walk
import os.path
import numpy
import scipy.io.wavfile as wavfile
import librosa.display
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import os
from werkzeug.utils import secure_filename
from flask import Flask
from flask import render_template, request
app = Flask(__name__)


# Import the libraries

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'wav', 'png', 'jpg', 'jpeg'}
#pp = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return "hello"
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return 'No file selected'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            signal, rate = librosa.load(UPLOAD_FOLDER+'/'+filename)
            # The Mel Spectrogram
            S = librosa.feature.melspectrogram(
                signal, sr=rate, n_fft=2048,    hop_length=512, n_mels=128)
            S_DB = librosa.power_to_db(S, ref=np.max)
            S_DB = S_DB.flatten()[:1200]
            clf = pickle.load(open('SVM.pkl', 'rb'))
            ans = clf.predict([S_DB])[0]
            music_class = str(ans)
            print(music_class)
            return music_class


@app.route('/uploadervgg', methods=['GET', 'POST'])
def classify_vgg():
    if request.method == 'GET':
        return "hello"
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return 'No file selected'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print( os.popen("pwd").read() )
            print("______________")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            base_model = VGG19(weights='imagenet')
            model = Model(inputs=base_model.input,
                          outputs=base_model.get_layer('flatten').output)
            image = load_img(UPLOAD_FOLDER+'/'+filename,
                             target_size=(224, 224, 3))
            np.expand_dims(image, axis=0)
            image = img_to_array(image)
            image = image.reshape(
                (1, image.shape[0], image.shape[1], image.shape[2]))

            image = preprocess_input(image)
            yhat = model.predict(image)
    # create a list containing the class labels
            class_labels = ["blues", "classical", "country",
                            "disco", "hiphop", "metal", "pop", "reggae", "rock"]
    # find the index of the class with maximum score
            pred = np.argmax(class_labels, axis=-1)
    # print the label of the class with maximum score
            return class_labels[pred]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000)
