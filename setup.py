from setuptools import setup, find_packages
import os

version = {}
with open(os.path.join("multivit", "version.py")) as fp:
    exec(fp.read(), version)

setup(
    name="multivit",
    version=version["__version__"],
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'multivit-training=multivit.bin.training:main',
            # Add more scripts here
        ],
    },
)
