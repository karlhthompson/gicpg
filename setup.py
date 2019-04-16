import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gicpg",
    version="1.0.1",
    author="Karl H. Thompson",
    author_email="karlht2@illinois.edu",
    description="A Package for generating and identifying patterns in large-scale networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/karlhthompson/gicpg",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)