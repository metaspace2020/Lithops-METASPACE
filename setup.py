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
          "pywren-ibm-cloud>=1.5.3",
          # pandas version should match the version in the runtime to ensure data generated locally can be unpickled
          # in pywren actions
          "pandas==0.25.1",
          "scipy==1.3.1",
          "pyImagingMSpec==0.1.4",
          "cpyImagingMSpec==0.3.2",
          "pyMSpec==0.1.2",
          "cpyMSpec==0.3.5",
          "pyimzML==1.3.0",
          "requests==2.22.0",
          "msgpack==0.6.2",
          "msgpack-numpy==0.4.4.3",
          "metaspace2020", # Only needed for experiment notebooks
          "psutil", # Only needed for experiment notebooks
      ])
