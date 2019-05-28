import sys
from setuptools import setup, find_packages

from annotation_pipeline import __version__

assert (3, 6) <= sys.version_info < (3, 7), 'Python 3.6 is required'

setup(name='pywren-annotation-pipeline',
      version=__version__,
      description='PyWren-powered METASPACE annotation pipeline',
      url='https://github.com/metaspace2020/pywren-annotation-pipeline',
      packages=find_packages(),
      install_requires=[
          "pywren-ibm-cloud>=1.0.8",
          # pandas version should match the version in the runtime to ensure data generated locally can be unpickled
          # in pywren actions
          "pandas==0.23.3",
          "pyImagingMSpec==0.1.4",
          "cpyImagingMSpec==0.3.2",
          "pyMSpec==0.1.2",
          "cpyMSpec==0.4.2",
          "pyimzML==1.2.5",
          "requests",
          "msgpack-numpy"
      ])
