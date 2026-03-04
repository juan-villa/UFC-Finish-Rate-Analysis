PYTHON ?= python3

setup:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

scrape:
	$(PYTHON) -m src.scraping

analyze:
	jupyter notebook notebooks/analysis.ipynb
