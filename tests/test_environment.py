

import filecmp

from utils_testing.utils_testing import path_repo_root


def test_environment():
    """
    Checks that the environment file `[repo]/environement.yml` matches the lock-file used in `make install` for creating our environment.

    When we run `make update-lock-file` we will create a conda-lock file (`[repo]/conda-linux-64.lock`) and then make a copy of
    `[repo]/environement.yml` to `[repo]/tests/data/copy_lock_filed_environment.yml`.

    If `[repo]/environement.yml` and `[repo]/tests/data/copy_lock_filed_environment.yml` are not equal we expected the lock file to be
    outdated.
    """
    path_current_env_file = path_repo_root() / "environment.yml"
    path_env_file_locked = path_repo_root() / "tests" / "data" / "copy_lock_filed_environment.yml"

    is_env_updated = filecmp.cmp(path_current_env_file, path_env_file_locked)
    assert is_env_updated, ("The 'environment.yml'-file have been updated and you need to update the environement lock-file. \n"
                            f"Run 'make env-create-lock-file' to ensure {path_current_env_file} matches our env lock"
                            "file 'conda-linux-64.lock'")
