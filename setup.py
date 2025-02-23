# setup.py
from setuptools import setup, find_packages

setup(
    name="data_analysis",
    version="0.1.0",
    packages=find_packages(),  # Automatically finds the my_data_tool/ folder
    install_requires=[
        "pandas",
        "numpy",
        "seaborn",
        "matplotlib",
        "scipy",
        "scikit-learn"
    ],
    python_requires=">=3.8",
    description="A simple data analysis toolkit",
    author="Nelson Alex",
    author_email="nelsontharappel@gmail.com",
)
