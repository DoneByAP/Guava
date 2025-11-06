from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README if it exists
readme_path = Path(__file__).parent / "README.md"
long_description = (
    readme_path.read_text(encoding="utf-8")
    if readme_path.exists()
    else "Guava: Distributed energy-aware neural network training across multiple GPUs / machines."
)

setup(
    name="guava",
    version="0.7.3",
    author="Guava Team",
    description="Distributed neural network training across multiple GPUs and machines with energy telemetry (data parallel, model parallel, pipeline parallel, tensor parallel).",
    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=find_packages(),
    python_requires=">=3.8",

    install_requires=[
        "numpy",
        "tqdm",
        "psutil",
        "requests; platform_system == 'Windows'",   # auto-OHM download helper
        "wmi; platform_system == 'Windows'",        # ✅ Windows CPU sensor bridge
        "pynvml; platform_system != 'Darwin'",
        "nvidia-ml-py; platform_system != 'Darwin'",  # NVML fallback
    ],

    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "telemetry": [
            "pynvml",
            "nvidia-ml-py",
            "requests",
            "wmi",
        ],
    },

    entry_points={
        "console_scripts": [
            # You can re-enable these once the scripts/ dir lives in package
            # "guava-orchestrator=guava.scripts.orchestrator_train:main",
            # "guava-worker=guava.scripts.guava_worker:main",
            "guava-version=guava.cli:main",
        ],
    },

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Monitoring",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
