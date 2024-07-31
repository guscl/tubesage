FROM python:3.11-slim

COPY . /app

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN pip install --no-cache-dir poetry

RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

EXPOSE 5000

ENTRYPOINT ["python"]

CMD ["tubesage/app.py"]