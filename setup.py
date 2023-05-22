"""Python setup.py for torch_bricks package"""
import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("torch_bricks", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="torch_bricks",
    version=read("torch_bricks", "VERSION"),
    description="Awesome torch_bricks created by PeteHeine",
    url="https://github.com/PeteHeine/torch_bricks/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="PeteHeine",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["torch_bricks = torch_bricks.__main__:main"]
    },
    extras_require={"test": read_requirements("requirements-test.txt")},
)
