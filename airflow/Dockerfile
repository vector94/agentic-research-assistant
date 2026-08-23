FROM apache/airflow:slim-3.3.0-python3.12

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        libpq-dev \
        poppler-utils \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=airflow:root airflow/requirements-airflow.txt /tmp/requirements-airflow.txt

USER airflow

RUN uv pip install \
    --python /home/airflow/.local/bin/python \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "apache-airflow==3.3.0" \
    -r /tmp/requirements-airflow.txt \
    && python -m pip check

ENV PYTHONPATH=/opt/airflow
