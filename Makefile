.ONESHELL:
PYTHONPATH=PYTHONPATH=$(shell pwd)/src

.PHONY: help
help:             ## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep


.PHONY: show
show:             ## Show the current environment.
	@echo "Current environment:"
	@mamba info

.PHONY: lint
lint:             ## Perform linting on all files using pre-commit
	pre-commit run --all-files

.PHONY: install
install:          ## create environment using lock-file
	@mamba create --name torch_bricks --file conda-linux-64.lock


.PHONY: update
update:           ## Update lock file using the specification in 'environment.yml'
	@conda-lock -k explicit --conda mamba -f environment.yml

.PHONY: test
test:        	  ## Run tests and generate coverage report.
	set -e
	$(PYTHONPATH) pytest -v --cov-config .coveragerc --cov=src -l --tb=short --maxfail=1 tests/
	coverage xml
	coverage html


.PHONY: watch
watch:            ## Run tests on every change.
	ls **/**.py | entr $(ENV_PREFIX)pytest -s -vvv -l --tb=long --maxfail=1 tests/

.PHONY: clean
clean:            ## Clean unused files.
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

# .PHONY: release
# release:          ## Create a new tag for release.
# 	@echo "WARNING: This operation will create s version tag and push to github"
# 	@read -p "Version? (provide the next x.y.z semver) : " TAG
# 	@echo "$${TAG}" > torch_bricks/VERSION
# 	@$(ENV_PREFIX)gitchangelog > HISTORY.md
# 	@git add torch_bricks/VERSION HISTORY.md
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
