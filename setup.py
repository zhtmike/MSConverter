from setuptools import setup

setup(
    name='ms-converter',
    version='0.1.0',
    description='MS Converter',
    author='Cheung Ka Wai',
    author_email='zhtmike@gmail.com',
    packages=['ms_converter'],
    data_files=[("assets",
                 ['assets/mapping.json', "assets/mindspore_v2.5.0.mint.rst"])],
    entry_points={
        'console_scripts': [
            'torch2mint = ms_converter.torch2mint:main',
        ],
    },
)
