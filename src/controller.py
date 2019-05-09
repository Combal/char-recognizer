from flask import Flask, request, render_template
from src.recognizer import Recognizer
import src.image_reader as ir
import cv2

app = Flask(__name__)
temp_file = 'data/temp_file.jpg'
temp_file_mod = 'data/temp_file_mod.jpg'


@app.route('/', methods=['GET', 'POST'])
def recognize():
    if request.method == 'POST':
        with open(temp_file, 'wb') as f:
            f.write(request.data)
        img = ir.read_and_transform(temp_file)
        cv2.imwrite(temp_file_mod, img)
        return recognizer.recognize(temp_file_mod)
    else:
        return render_template("index.html")


if __name__ == '__main__':
    recognizer = Recognizer()
    app.run(host='0.0.0.0', port=8008, threaded=False)
