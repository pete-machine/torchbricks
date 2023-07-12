.ONESHELL:
PYTHONPATH=PYTHONPATH=$(shell pwd)/src:$(shell pwd)/tests

.PHONY: help
help:             ## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep


.PHONY: show
show:                    ## Show the current environment.
	@echo "Current environment:"
	@mamba info

.PHONY: lint
lint:                    ## Perform linting on all files using pre-commit
	pre-commit run --all-files

.PHONY: install
install:                 ## create environment using lock-file
	mamba create --name torchbricks --file conda-linux-64.lock


.PHONY: update-lock-file
update-lock-file:        ## Update lock file using the specification in 'environment.yml'
	conda-lock -k explicit --conda mamba -f environment.yml

.PHONY: update
update: update-lock-file install ## Update lock file using the specification in 'environment.yml'

.PHONY: test-all
test-all:        	         ## Run tests and generate coverage report.
	@set -e
	$(PYTHONPATH) pytest -v --cov-config .coveragerc --cov=src -l --tb=short --maxfail=1 --durations=0 tests/
	coverage xml
	coverage html


.PHONY: test
test:
	$(PYTHONPATH) pytest --durations=0 -m "not slow" tests/

.PHONY: train-cifar10
train-cifar10:           ## Run CIFAR10 training
	@set -e
	$(PYTHONPATH) python scripts/train_lightning_cifar10.py --batch_size 256 --num_workers 10 --max_epochs 20 --accelerator gpu

.PHONY: readme-update
readme-update:        ## Convert README.ipynb to README.md
	@jupyter nbconvert --clear-output --to notebook --output=build/tmp_readme.ipynb README.ipynb
	@jupyter nbconvert --to markdown --output=../README.md build/tmp_readme.ipynb
# PYTHONPATH=src jupyter nbconvert --execute --to markdown README.ipynb

.PHONY: watch
watch:                   ## Run tests on every change.
	ls **/**.py | entr $(ENV_PREFIX)pytest -s -vvv -l --tb=long --maxfail=1 tests/

.PHONY: clean
clean:                   ## Clean unused files.
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

# .PHONY: build
# build: clean		 	  ## Clean and build
# 	@set -e
# 	python -m build
# 	twine check dist/*
# 	# twine upload -r testpypi dist/*
# 	twine upload dist/*


.PHONY: release
release:          ## Create a new tag for release.
	bumpver update --patch --push

# bumpver update --patch
# git push origin 0.0.5

# 	@echo "WARNING: This operation will create s version tag and push to github"
# 	@read -p "Version? (provide the next x.y.z semver) : " TAG
# 	@echo "$${TAG}" > torchbricks/VERSION
# 	@$(ENV_PREFIX)gitchangelog > HISTORY.md
# 	@git add torchbricks/VERSION HISTORY.md
# 	@git commit -m "release: version $${TAG} 🚀"
# 	@echo "creating git tag : $${TAG}"
# 	@git tag $${TAG}
# 	@git push -u origin HEAD --tags
# 	@echo "Github Actions will detect the new tag and release the new version."

# .PHONY: docs
# docs:             ## Build the documentation.
# 	@echo "building documentation ..."
# 	@$(ENV_PREFIX)mkdocs build
# 	URL="site/index.html"; xdg-open $$URL || sensible-browser $$URL || x-www-browser $$URL || gnome-open $$URL
