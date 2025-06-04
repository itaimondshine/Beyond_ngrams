# Dockerfile
FROM robd003/python3.10
RUN apt-get update -y
RUN apt-get install -y python3-pip
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . /app_instructor
WORKDIR /app_instructor
ENTRYPOINT ["python"]
CMD ["app_instructor.py"]
