FROM python:3.8
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN apt-get update
RUN apt-get install -y npm

COPY package.json .
RUN npm install

COPY . .
RUN npm run build

ENV PORT 8000
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 mlg.wsgi:application
# CMD exec daphne mlg.asgi:application --bind 0.0.0.0 --port $PORT --verbosity 1
