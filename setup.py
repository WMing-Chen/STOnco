from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stride-spatial",
    version="1.0.0",
    author="STRIDE Team",
    author_email="weimingchen2025@163.com",  # 请替换为您的实际邮箱地址
    description="STRIDE: Spatial Transcriptomics Tumor Classification with Dual-Domain Adversarial Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChenWeiMing-lab/STRIDE-spatial-transcriptomics",  # 请替换为您的实际GitHub用户名
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "stride-train=stride.core.train:main",
            "stride-infer=stride.core.infer:main",
            "stride-prepare=stride.utils.prepare_data:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)