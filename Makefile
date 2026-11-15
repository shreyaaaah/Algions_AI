.PHONY: install data train test api dashboard
install: ; pip install -r requirements.txt
data:    ; python scripts/download_data.py
train:   ; python -m src.cinema_demand.train
test:    ; python -m pytest -q
api:     ; uvicorn app.api:app --reload --port 8000
dashboard: ; streamlit run app/dashboard.py
