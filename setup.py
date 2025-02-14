from setuptools import setup, find_packages

setup(
    name="borelliCDL",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "numba",
        "alphacsc",
        "pandas"
    ],
    author="Rémi Al Ajroudi",
    description="A convolutional dictionary learning library for signal processing in the context of large multivariate signals.",
    license="MIT",
)
