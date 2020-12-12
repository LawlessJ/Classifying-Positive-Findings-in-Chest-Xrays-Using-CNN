from tensorflow import keras
from PIL import Image
import numpy as np

try:
    print('''
    Please wait while the model loads...\n
    You will see loading text, followed by an input prompt for the file when the model is prepped.
    Do not type while the model loads or your input prompt will result in an error.
    ''')
    loaded_model = keras.models.load_model('binary')
except:
    modelpath = input('Hint: Did you unzip the file?\nPlease enter the relative path to your saved model file, including backslashes: ')
    print('Please wait while the model loads...')
    loaded_model = keras.models.load_model(modelpath)

path = input('Please enter the relative path to your xray, including backslashes: ')

with Image.open(path) as xray_img:
    xray = xray_img.resize((224, 224))

    cnn_input = np.asarray(xray) / 255
    cnn_input = np.repeat(cnn_input[..., np.newaxis], repeats = 3, axis = -1)

    cnn_input = cnn_input.reshape(1, 224, 224, 3)

    prediction = loaded_model.predict(cnn_input)[0][0]
    print(f'Predicted probability of a positive finding in this image: {prediction}')

    if prediction < 0.5:
        print('No Finding in this image')
    else:
        print('The model indicates a positive finding in this image')
