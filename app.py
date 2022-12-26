from flask import Flask, render_template, request
import os
import tensorflow as tf
from keras.models import load_model 
from keras.preprocessing import image
import numpy as np


image_folder = os.path.join('static', 'images')
app = Flask(__name__)
# run_with_ngrok(app=app)
app.config["UPLOAD_FOLDER"] = image_folder

# model = load_model('/content/fullmodel.h5')
from keras.models import load_model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# # load weights into a new model
# model.load_weights("model.h5")  
model = load_model('tbc_model.h5')
img_height = 180
img_width = 180
class_name= ['Normal', 'Tuberculosis']


@app.route('/', methods=['GET'])
def home():
  return render_template('index.html',user_image='static\images\hospital.png')

@app.route('/', methods=['GET','POST'])

def diagnosis():
    # Download image
    imagefile = request.files['imagefile']
    image_path = './static/images/' + imagefile.filename 
    imagefile.save(image_path)
    ##YOUR CODE GOES HERE##
    img = tf.keras.utils.load_img(
    image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
   
    ##YOUR CODE GOES HERE##
     # Show image
    ##YOUR CODE GOES HERE##
    # plt.figure()
    # plt.imshow(image)

    # Load model  
    ##YOUR CODE GOES HERE##

    # Predict the diagnosis
    ##YOUR CODE GOES HERE##
    predictions = model.predict(img_array)
    
    # Find the name of the diagnosis  
    ##YOUR CODE GOES HERE##
    diag = class_name[int(np.argmax(predictions, axis=1))]
    pic = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
    # return pic
    # return diag
    return render_template('index.html', user_image=pic, prediction_text=diag)
  
if __name__=='__main__':
  app.debug = True
  app.run()