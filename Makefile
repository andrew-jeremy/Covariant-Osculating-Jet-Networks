.PHONY: install test quickstart benchmark clean
install:
	python -m pip install -e '.[all]'

test:
	python -m pytest -q

quickstart:
	python examples/quickstart.py

benchmark:
	python benchmarks/jet_recovery.py --epochs 300 --seeds 3 --output results/jet_recovery.json

clean:
	rm -rf .pytest_cache build dist *.egg-info src/*.egg-info
