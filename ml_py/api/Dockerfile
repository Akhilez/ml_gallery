FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

COPY docker_requirements.txt .
RUN pip install -r docker_requirements.txt

COPY *requirements.txt ./
RUN if [ -f ./project_requirements.txt ] ; then pip install -r project_requirements.txt ; fi

COPY . /app
