.PHONY: setup run test format lint clean

setup:
	pip install -r requirements.txt
	pip install black isort flake8 pytest

run:
	python run_pipeline.py

test:
	pytest tests/

format:
	isort .
	black .

lint:
	flake8 src run_pipeline.py

clean:
	rm -rf __pycache__ .pytest_cache
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
