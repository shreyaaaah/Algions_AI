FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 8501
# run training once (needs data/raw/cinemaTicket_Ref.csv mounted), then serve:
#   docker run -v $PWD/data:/app/data -v $PWD/artifacts:/app/artifacts <image> python -m src.cinema_demand.train
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
