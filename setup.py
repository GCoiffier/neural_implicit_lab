import setuptools

with open("README.md", "r") as file_handle:
    long_description = file_handle.read()

setuptools.setup(
    name="implicitLab",
    version='0.1',
    author="GCoiffier",
    description="Neural implicit representations of geometry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-image",
        "mouette",
        "torch",
        "libigl",
        "triangle",
        "tqdm",
        "deel-torchlip"
    ]
)
