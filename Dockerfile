FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run the API on Hugging Face Spaces default port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
