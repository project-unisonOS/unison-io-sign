PYTHON ?= python3
VENV := .venv
ACTIVATE := $(VENV)/bin/activate

$(VENV):
	$(PYTHON) -m venv $(VENV)
	. $(ACTIVATE) && pip install -U pip

install: $(VENV)
	. $(ACTIVATE) && pip install -c constraints.txt -r requirements.txt

test: install
	. $(ACTIVATE) && PYTHONPATH=./src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache

.PHONY: install test clean
