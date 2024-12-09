from setuptools import find_packages, setup

setup(
    name="adaptive-attention-tiling",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy>=2.0.0",  # Updated to require numpy 2.x
        "pytest",
        "mlflow",
        "matplotlib",
        "transformers",
    ],
)
