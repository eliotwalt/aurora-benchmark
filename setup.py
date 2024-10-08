from setuptools import setup, find_packages

setup(
    name='aurora-benchmark',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # Add your package dependencies here
        "ipykernel",
        "cftime",
        "netcdf4",
        "h5netcdf",
        "dask",
        "xarray[complete]",
        "microsoft-aurora",
        "gcsfs",
        "PyYAML",
        "cartopy",
    ],
    python_requires='>=3.10',
    author='Eliot Walt',
    author_email='walt.eliot@hotmail.com',
    description='A package to evaluate Microsoft\'s Aurora on reanalysis data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/eliotwalt/aurora-benchmark',
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)