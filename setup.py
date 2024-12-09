from setuptools import setup, find_packages

setup(
    name="adaptive-attention-tiling",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy<2.0.0",  # Pin numpy to 1.x for torch compatibility
        "pytest",
        "mlflow",
        "matplotlib",
        "transformers",
    ],
)
