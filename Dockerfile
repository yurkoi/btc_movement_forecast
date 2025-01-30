FROM python:3.10

WORKDIR /app

COPY requirements_web.txt /app/
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements_web.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8000"]
