# char-recognizer

char recognizer for georgian letters, implemented on python using tensorflow based on MNIST example

## Requirements

python 3.6

- numpy
- flask>=0.12.2
- flask_restplus>=0.10.1
- requests>=2.18.4
- Pillow>=4.1.1
- keras
- opencv-python<4
- tensorflow


## Run

### run keras trainer
create directory `data/train_images/` like this:

```
+ data
    + train_images
        + ა
            - file1.jpg
            - file2.jpg
            - ...
        + ბ
            - ...
        + გ

```

```
python -m src.keras_trainer
```

### run web application

```
python -m src.controller
```

then visit http://127.0.0.1:8008