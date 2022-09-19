import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neural_clamping",
    version="0.0.1",
    author="Yung-Chen Tang",
    author_email="yctang@m109.nthu.edu.tw",
    description="Neural Clamping",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yungchentang/neural-clamping",
    packages=setuptools.find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'matplotlib',
        'requests',
        'numpy',
        'tqdm',
        'scipy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)