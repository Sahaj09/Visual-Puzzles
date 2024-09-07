import os
import zipfile
from setuptools import setup, find_packages
from setuptools.command.install import install

class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        # Unzip the data package after installation
        data_dir = os.path.join(self.install_lib, 'visual_puzzle', 'assets') # type: ignore
        zip_path = os.path.join(data_dir, 'rush.txt.zip')
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            os.remove(zip_path)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="Visual-Puzzles",
    version="0.1.0",
    author="Sahaj Singh Maini",
    author_email="sahmaini@iu.edu",
    description="A environment for visual puzzles to evaluate VLMs and other models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sahaj09/Visual-Puzzles",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "visual_puzzle": ["assets/*"],
    },
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    cmdclass={
        'install': CustomInstallCommand,
    },
)