# How to develop on this project

torchbricks welcomes contributions from the community.

This instructions are for linux base systems.
## Setting up your own fork of this repo.

- On github interface click on `Fork` button.
- Clone your fork of this repo. `git clone git@github.com:YOUR_GIT_USERNAME/torchbricks.git`
- Enter the directory `cd torchbricks`
- Add upstream repo `git remote add upstream https://github.com/pete-machine/torchbricks`

## Install the project in develop mode

Run `make install` to install the project in develop mode.

## Run the tests to ensure everything is working

Run `make test` to run the tests.

## Create a new branch to work on your contribution

Run `git checkout -b my_contribution`

## Make your changes

Edit the files using your preferred editor. (we recommend VSCode)

## Run the linter

Run `make lint` to run the linter.

## Test your changes

Run `make test` to run the tests.

Ensure code coverage is high and add tests to your PR.

## Commit your changes

This project uses [conventional git commit messages](https://www.conventionalcommits.org/en/v1.0.0/).

Example: `fix(package): update setup.py arguments 🎉` (emojis are fine too)

## Push your changes to your fork

Run `git push origin my_contribution`

## Submit a pull request

On github interface, click on `Pull Request` button.

Wait CI to run and one of the developers will review your PR.
## Makefile utilities

This project comes with a `Makefile` that contains a number of useful utility.

```bash
❯ make
clean                     Clean unused files.
install                   create environment using lock-file
lint                      Perform linting on all files using pre-commit
readme-update             Convert README.ipynb to README.md
release-minor             Create a new minor (X.Y+1.Z) tag for release.
release-patch             Create a new patch (X.Y.Z+1) tag for release.
show                      Show the current environment.
test-all                  Run tests and generate coverage report.
update-lock-file          Update lock file using the specification in 'environment.yml'
update                    Update lock file and environment using the specification in 'environment.yml'
```
