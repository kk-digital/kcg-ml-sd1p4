FROM docker.io/nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

WORKDIR /app

COPY app.py /app/app.py
COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -r requirements.txt

VOLUME ["/host/input", "/host/output"]

ENV FLASK_APP=app.py

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0"]
