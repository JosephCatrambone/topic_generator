FROM python:3.7.7-slim
COPY requirements.txt /
RUN pip3 install -r /requirements.txt
COPY . /app
WORKDIR /app
RUN python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/byt5-small')"
ENTRYPOINT ["./serve_production.sh"]