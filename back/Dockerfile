FROM tensorflow/tensorflow

EXPOSE 6000

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /back
COPY . /back

ENV NUMBA_CACHE_DIR=/tmp/numba_cache

#CMD ["cat","app.py"]

CMD ["python","app.py"]