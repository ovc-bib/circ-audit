"""
setup.py — CircAudit package setup
====================================
Makes CircAudit pip-installable for open-source release.
"""

from setuptools import setup, find_packages

setup(
    name="circ-audit",
    version="0.1.0",
    description="Circularity-Aware Audit Tool for Biomedical Prediction",
    long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Cui Lei",
    author_email="cuilei@sj-hospital.org",
    url="https://github.com/cuilei-bib/circ-audit",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.3",
        "scipy>=1.7",
        "scikit-learn>=1.0",
        "matplotlib>=3.4",
    ],
    extras_require={
        "xgboost": ["xgboost>=1.5"],
        "plotly": ["plotly>=5.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
