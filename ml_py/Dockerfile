# FROM pytorch
FROM python:3.8
RUN pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
WORKDIR /PyMLGallery
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV PORT 8000
# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 ml_py.wsgi:application
CMD exec daphne ml_py.asgi:application --bind 0.0.0.0 --port $PORT --verbosity 1
