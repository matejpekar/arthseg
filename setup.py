import numpy
import os
from glob import glob
from pathlib import Path
from setuptools import Extension, setup
from wheel.bdist_wheel import bdist_wheel


class bdist_wheel_abi3(bdist_wheel):
    def get_tag(self):
        python, abi, plat = super().get_tag()

        if python.startswith("cp"):
            # on CPython, our wheels are abi3 and compatible back to 3.6
            return "cp38", "abi3", plat

        return python, abi, plat


# platform specific settings
if os.name == 'nt':
    flags = ['/std:c++17']
else:
    flags = ['-std=c++17']

module = Extension(
    'arthseg',
    sources=glob('arthseg/**/*.cpp', recursive=True),
    include_dirs=[numpy.get_include(), 'arthseg', 'arthseg/lib'],
    extra_compile_args=flags,
    define_macros=[("Py_LIMITED_API", "0x03060000")],
    py_limited_api=True,
)

setup(
    name='arthseg',
    version='0.1.3',
    license='MIT',
    description='Native library for arthropod segmentation',
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type='text/markdown',
    author='Matej Pekar',
    author_email='matej.pekar120@gmail.com',
    url='https://github.com/matejpekar/arthseg',
    install_requires=['numpy'],
    ext_modules=[module],
    package_data={'arthseg': ['__init__.pyi']},
    packages=['arthseg'],
    cmdclass={"bdist_wheel": bdist_wheel_abi3},
)
