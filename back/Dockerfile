FROM python:3.8

EXPOSE 6000

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /back
COPY . /back

ENV NUMBA_CACHE_DIR=/tmp/numba_cache

RUN useradd appuser && chown -R appuser /app
USER appuser

CMD ["python","app.py"]