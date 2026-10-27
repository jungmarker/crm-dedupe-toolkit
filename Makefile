PY=.venv/bin

setup:
	python3 -m venv .venv && $(PY)/pip install -q -e ".[dev]"

data:
	$(PY)/ledgerly-generate --seed 42 --out data/seed42
	$(PY)/ledgerly-generate --seed 7 --out data/seed7

train:
	$(PY)/crm-dedupe train --data data/seed42 --labels 400

run:
	$(PY)/crm-dedupe run --data data/seed7 --out out/seed7

test:
	$(PY)/pytest -q

all: data train run test

.PHONY: setup data train run test all
