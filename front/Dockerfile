FROM python:3.8

EXPOSE 7000

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /front
COPY . /front

CMD ["python", "app.py"]
