.ONESHELL:
PYTHONPATH=PYTHONPATH=$(shell pwd)/src:$(shell pwd)/tests


## Auto-documented makefile: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

.PHONY: show
show: ## Show the current environment.
	@echo "Current environment:"
	@micromamba info

.PHONY: lint
lint: ## Perform linting on all files using pre-commit
	pre-commit run --all-files

.PHONY: env-install
env-install: ## create environment using lock-file
	micromamba create --name torchbricks --file conda-linux-64.lock

.PHONY: env-install-min
env-install-min: ## create environment using lock-file
	micromamba create --name torchbricks_low --file environment-min.yml

.PHONY: env-create-lock-file
env-create-lock-file: ## Update lock file using the specification in 'environment.yml'
	@set -e
	conda-lock -k explicit --conda micromamba -f environment.yml
	cp environment.yml tests/data/copy_lock_filed_environment.yml


.PHONY: env-create-lock-and-install
env-create-lock-and-install: env-create-lock-file env-install ## Update lock file and environment using the specification in 'environment.yml'

.PHONY: test-all
test-all: ## Run tests and generate coverage report.
	@set -e
	$(PYTHONPATH) pytest -v --cov-config .coveragerc --cov=src -l --tb=short --maxfail=1 --durations=0 tests/
	coverage xml
	coverage html


.PHONY: test
test:
	$(PYTHONPATH) pytest --durations=0 -m "not slow" tests/

.PHONY: train-cifar10
train-cifar10: ## Run CIFAR10 training
	@set -e
	$(PYTHONPATH) python scripts/train_lightning_cifar10.py --batch_size 256 --num_workers 10 --max_epochs 20 --accelerator gpu

.PHONY: readme-update
readme-update: ## Convert README.ipynb to README.md
	@jupyter nbconvert --clear-output --to notebook --output=build/tmp_readme.ipynb README.ipynb
	@jupyter nbconvert --to markdown --output=../README.md build/tmp_readme.ipynb

.PHONY: clean
clean:  ## Clean unused files.
	@find ./ -name '*.pyc' -exec rm -f {} \;
	@find ./ -name '__pycache__' -exec rm -rf {} \;
	@find ./ -name 'Thumbs.db' -exec rm -f {} \;
	@find ./ -name '*~' -exec rm -f {} \;
	@rm -rf .cache
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -rf htmlcov
	@rm -rf .tox/
	@rm -rf docs/_build


.PHONY: release-patch
release-patch: ## Create a new patch (X.Y.Z+1) tag for release.
	bumpver update --patch --push

.PHONY: release-minor
release-minor: ## Create a new minor (X.Y+1.Z) tag for release.
	bumpver update --minor --push
