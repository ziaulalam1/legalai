.PHONY: fmt lint test build run worker clean

fmt:
	python -m pip install -q ruff || true
	ruff format .

lint:
	python -m pip install -q ruff || true
	ruff check .

test:
	python -m pip install -q pytest || true
	pytest -q

build:
	docker compose build

run:
	docker compose run --rm legalai ldc --help

worker:
	docker compose up

clean:
	docker compose down --remove-orphans || true
	docker image rm -f legalai:local || true
	rm -rf runtime/outbox/* runtime/inbox/* || true
