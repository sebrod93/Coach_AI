FROM python:3.8-slim
RUN mkdir /app
WORKDIR /app
COPY . /app
RUN pip install -U .
CMD uvicorn api.fast:app --host 0.0.0.0 --port 8080
COPY /home/lnguyen/code/klisangn/gcp/wagon-bootcamp-319219-531b9273b5fe.json /credentials.json
