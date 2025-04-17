.ONESHELL:
PYTHONPATH=PYTHONPATH=$(shell pwd)/src:$(shell pwd)/tests


## Auto-documented makefile: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
# https://mmngreco.dev/posts/uv-makefile/

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

.PHONY: lint
lint: ## Perform linting on all files using pre-commit
	uv run --frozen pre-commit run --all-files

.PHONY: test-all
test-all: ## Run tests and generate coverage report.
	@set -e
	$(PYTHONPATH) uv run --frozen pytest -v --cov-config .coveragerc --cov=src -l --tb=short --maxfail=1 --durations=0 tests/
	coverage xml
	coverage html


.PHONY: test
test:
	$(PYTHONPATH) uv run --frozen pytest --durations=0 -m "not slow" tests/

.PHONY: train-cifar10
train-cifar10: ## Run CIFAR10 training
	@set -e
	$(PYTHONPATH) python scripts/train_lightning_cifar10.py --batch_size 256 --num_workers 10 --max_epochs 20 --accelerator cuda

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
