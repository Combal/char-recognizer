from flask import Flask, request, render_template
from src.recognizer import Recognizer
from src import image_utils
import cv2
import numpy as np

app = Flask(__name__)
temp_file = 'data/temp_file.jpg'
temp_file_mod = 'data/temp_file_mod.jpg'


@app.route('/', methods=['GET', 'POST'])
def recognize():
    if request.method == 'POST':
        with open(temp_file, 'wb') as f:
            f.write(request.data)
        img = cv2.imdecode(np.asarray(bytearray(request.data), dtype="uint8"), cv2.IMREAD_GRAYSCALE)
        img = image_utils.transform(img)
        cv2.imwrite(temp_file_mod, img)
        return recognizer.recognize(img)
    else:
        return render_template("index.html")


if __name__ == '__main__':
    recognizer = Recognizer()
    app.run(host='0.0.0.0', port=8008, threaded=False)
