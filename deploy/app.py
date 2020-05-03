#::: Import modules and packages :::
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
# Import Keras dependencies
from keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
ops.reset_default_graph()
import random
import h5py
from PIL import Image
import PIL
from vb100_utils import *
##################################################
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
ops.reset_default_graph()
from keras.preprocessing import image
# Import other dependecies
import numpy as np
import h5py
from PIL import Image
import PIL
import os
#import model_select
import requests
##########################################
#::: Flask App Engine :::
# Define a Flask app
app = Flask(__name__)
# ::: FLASK ROUTES
@app.route('/', methods=['GET'])
def index():
	# Main Page
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
	a = ''
	classes = {'TRAIN': ['COVID', 'NORMAL', 'NORMALP'],
			   'VALIDATION': ['COVID', 'NORMAL', 'NORMALP'],
			   'TEST': ['COVID', 'NORMAL', 'NORMALP']}

	print('h5py version is {}'.format(h5py.__version__))

	# Get the architecture of CNN
	json_file = open('model_adam_20191030_01.json')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	# Get weights into the model
	loaded_model.load_weights('model_100_eopchs_adam_20191030_01.h5')

	# Define optimizer and run
	opt = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
	loaded_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')

	if request.method == 'POST':
		test_data = [0, 1]
		prediction_value = random.choice(test_data)
				# Get the file from post request
		f = request.files['file']
		# Save the file to ./uploads
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
		f.save(file_path)

		IMG = Image.open(file_path)
		print(type(IMG))
		IMG = IMG.resize((333, 250))
		IMG = np.array(IMG)
		print('po array = {}'.format(IMG.shape))
		IMG = np.true_divide(IMG, 255)
		IMG = IMG.reshape(4, 333, 250, 1)
		print(type(IMG), IMG.shape)

		predictions = loaded_model.predict(IMG)
		predictions_c = loaded_model.predict_classes(IMG)
		predicted_class = classes['TRAIN'][prediction_value]
		return str(predicted_class)

@app.route("/news", methods=['GET', 'POST'])
def news():
	main_url = "http://newsapi.org/v2/top-headlines?country=in&category=health&apiKey=dbf949d1eea745a6a83a4d45c62064*4**"

	# fetching data in json format
	open_bbc_page = requests.get(main_url).json()

	# getting all articles in a string article
	article = open_bbc_page["articles"]

	# empty list which will
	# contain all trending news
	results = []
	results_link = []
	for ar in article:
		results.append(ar["title"])
		results_link.append(ar["url"])

	return render_template('sucess.html', results=results, results_link=results_link)

@app.route("/chat")
def chat():
	return render_template('chat.html')

if __name__ == '__main__':
	app.run(debug = True)
