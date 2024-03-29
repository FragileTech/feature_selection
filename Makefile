current_dir = $(shell pwd)

PROJECT = feature_selection
n ?= auto
DOCKER_ORG = fragiletech
DOCKER_TAG ?= ${PROJECT}
VERSION ?= latest

.POSIX:
style:
	python3 -m black .
	python3 -m isort .

.POSIX:
check:
	!(grep -R /tmp tests)
	flakehell lint ${PROJECT}
	pylint ${PROJECT}
	black --check ${PROJECT}

.PHONY: test
test:
	find -name "*.pyc" -delete
	pytest -n $n -s -o log_cli=true -o log_cli_level=info

.PHONY: test-codecov
test-codecov:
	find -name "*.pyc" -delete
	pytest -n $n -s -o log_cli=true -o log_cli_level=info --cov=./ --cov-report=xml --cov-config=pyproject.toml

.PHONY: pipenv-install
pipenv-install:
	rm -rf *.egg-info && rm -rf build && rm -rf __pycache__
	rm -f Pipfile && rm -f Pipfile.lock
	pipenv install --dev -r requirements-test.txt
	pipenv install --pre --dev -r requirements-lint.txt
	pipenv install -r requirements.txt
	pipenv install -e .
	pipenv lock

.PHONY: pipenv-test
pipenv-test:
	find -name "*.pyc" -delete
	pipenv run pytest -s

.PHONY: docker-shell
docker-shell:
	docker run --rm --gpus all -v ${current_dir}:/${PROJECT} --network host -w /${PROJECT} -it ${DOCKER_ORG}/${PROJECT}:${VERSION} bash

.PHONY: docker-notebook
docker-notebook:
	docker run --rm --gpus all -v ${current_dir}:/${PROJECT} --network host -w /${PROJECT} -it ${DOCKER_ORG}/${PROJECT}:${VERSION}

.PHONY: docker-build
docker-build:
	docker build --pull -t ${DOCKER_ORG}/${PROJECT}:${VERSION} .

.PHONY: docker-test
docker-test:
	find -name "*.pyc" -delete
	docker run --rm --network host -w /${PROJECT} --entrypoint python3 ${DOCKER_ORG}/${PROJECT}:${VERSION} -m pytest -n $n -s -o log_cli=true -o log_cli_level=info

.PHONY: docker-push
docker-push:
	docker push ${DOCKER_ORG}/${DOCKER_TAG}:${VERSION}
	docker tag ${DOCKER_ORG}/${DOCKER_TAG}:${VERSION} ${DOCKER_ORG}/${DOCKER_TAG}:latest
	docker push ${DOCKER_ORG}/${DOCKER_TAG}:latest
