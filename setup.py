from setuptools import setup, find_packages

setup(
    name="adaptive-attention-tiling",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pytest",
        "mlflow",
        "matplotlib",
        "transformers",
    ],
)
