FROM python:3.6

WORKDIR /chars

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8008

COPY data/models/model-1557393970.7916455.h5 data/model.h5

COPY src src

CMD ["python", "-m", "src.controller"]