.PHONY: all data validate train evaluate test clean

all: data validate train evaluate

data:
	python src/simulate_dataset.py --config config/config.yaml

validate:
	python src/validate_dataset.py --config config/config.yaml

test:
	python -m pytest -q

train:
	python src/train_pipeline.py --config config/config.yaml

evaluate:
	python src/evaluate.py --config config/config.yaml

clean:
	rm -rf models/* reports/* data/processed/* data/raw/*.csv.gz
