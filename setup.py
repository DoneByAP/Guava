from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README if it exists, otherwise fallback
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else """
Guava: distributed neural network training across multiple GPUs / machines.
"""

setup(
    name="guava",
    version="0.1.0",
    author="Azani",
    description="Distributed neural network training across multiple GPUs and machines (data parallel, model parallel, pipeline parallel).",
    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=find_packages(),  # <- finds 'guava'
    python_requires=">=3.8",

    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "tqdm>=4.60.0",
        "psutil>=5.8.0",
    ],

    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
