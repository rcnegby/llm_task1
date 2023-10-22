FROM python:3.9.7

WORKDIR /app
ADD requirements.txt requirements.txt
ADD models models
ADD data data
RUN python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install -r requirements.txt
ADD main.py main.py


CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]

