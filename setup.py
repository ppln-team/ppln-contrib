import codecs
import os
import re

from setuptools import find_packages, setup


def get_absolute_path(*args):
    """Transform relative pathnames into absolute pathnames."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *args)


def get_contents(*args):
    """Get the contents of a file relative to the source distribution directory."""
    with codecs.open(get_absolute_path(*args), "r", "UTF-8") as handle:
        return handle.read()


def get_version(*args):
    """Extract the version number from a Python module."""
    contents = get_contents(*args)
    metadata = dict(re.findall("__([a-z]+)__ = ['\"]([^'\"]+)", contents))
    return metadata["version"]


setup(
    name="ppln-contrib",
    version=get_version("ppln_contrib", "__init__.py"),
    author="Miras Amir",
    author_email="amirassov@gmail.com",
    description="Extenstions for ppln",
    long_description_content_type="text/markdown",
    url="https://github.com/ppln-team/ppln-contrib",
    packages=find_packages(),
    scripts=[
        "scripts/torch_launch.sh"
    ],
    extras_require={"testing": ["pytest", "flake8", "black==19.3b0", "isort"]},
    python_requires=">=3.6.0",
)