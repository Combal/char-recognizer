FROM python:2.7

WORKDIR /chars

COPY requirements.txt /chars/requirements.txt

RUN pip install -r /chars/requirements.txt

COPY data/model.h5 /chars/data/model.h5

COPY src /chars/src

EXPOSE 8008

ENTRYPOINT ["python", "-m", "src/controller"]