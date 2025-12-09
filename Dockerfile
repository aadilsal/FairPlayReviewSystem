FROM python:3.11-slim
WORKDIR /app


RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt ./requirements.txt
COPY mlops/requirements.txt ./mlops-requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r mlops-requirements.txt


COPY . /app

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "mlops.server:app", "--host", "0.0.0.0", "--port", "8000"]
