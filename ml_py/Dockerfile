FROM python:3.8
WORKDIR /NNG
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV PORT 8000
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 nng.wsgi:application
