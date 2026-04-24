.PHONY: test bench bench-compare lint type cov check notebooks images docs clean

test:
	pytest -q

bench:
	pytest -m benchmark --benchmark-only --benchmark-autosave

bench-compare:
	pytest -m benchmark --benchmark-only \
	    --benchmark-compare --benchmark-compare-fail=mean:20%

lint:
	ruff check src tests

type:
	mypy src

cov:
	pytest --cov --cov-report=term-missing --cov-report=xml

notebooks:
	pytest --nbmake examples/

check: lint type cov

images:
	python3 docs/generate_images.py

docs: images

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml \
	       .benchmarks src/harmonyemissions.egg-info \
	       $(shell find . -type d -name __pycache__)
