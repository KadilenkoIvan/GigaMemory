FROM cr.ai.cloud.ru/aicloud-base-images/cuda11.8-torch2-py310:0.0.36 

USER root

WORKDIR /app 

COPY models/GigaChat-20B-A3B-instruct-v1.5-bf16 models/GigaChat-20B-A3B-instruct-v1.5-bf16 

COPY requirements.txt tmp/requirements.txt 
RUN pip install --timeout 120 -r tmp/requirements.txt

USER jovyan
WORKDIR /home/jovyan