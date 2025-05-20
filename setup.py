import glob
from setuptools import setup

setup(
    name="ms-converter",
    version="0.1.0",
    description="MS Converter",
    long_description=
    "MS Converter can replace PyTorch interfaces in scripts with MindSpore (>=2.5) Mint interfaces",
    author="Cheung Ka Wai",
    author_email="zhtmike@gmail.com",
    packages=["ms_converter"],
    data_files=[("ms_converter/assets",
                 ["assets/mapping.json",
                  *glob.glob("assets/mindspore_*.rst")])],
    entry_points={
        "console_scripts": [
            "torch2mint = ms_converter.torch2mint:main",
        ],
    },
    python_requires='>=3.9',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    license="MIT",
)
