from setuptools import setup, find_namespace_packages

setup(
    name="tp2-sia",
    version="0.1.0",
    package_dir={"": ".."},
    packages=["tp2", "tp2.src", "tp2.src.models", "tp2.src.engine", "tp2.src.operators", "tp2.src.stopping", "tp2.src.metrics"],
    entry_points={
        "console_scripts": [
            "image-ga = tp2.src.cli:main",
        ],
    },
)
