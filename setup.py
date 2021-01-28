from setuptools import setup, find_packages


setup(name='Lithops-METASPACE',
      version='1.0.0',
      description='Lithops-powered METASPACE annotation pipeline',
      url='https://github.com/metaspace2020/Lithops-METASPACE',
      packages=find_packages(),
      install_requires=[
          "lithops==2.2.16",
          # pandas version should match the version in the runtime to ensure data generated locally can be unpickled
          # in Lithops actions
          "pandas==1.1.3",
          "pyarrow==1.0.1",
          "scipy==1.5.3",
          "pyImagingMSpec==0.1.4",
          "cpyImagingMSpec==0.2.4",
          "pyMSpec==0.1.2",
          "cpyMSpec==0.3.5",
          "pyimzML==1.4.1",
          "requests==2.22.0",
          "msgpack==0.6.2",
          "msgpack-numpy==0.4.4.3",
          "pypng==0.0.19",  # 0.0.20 introduced incompatible API changes
          "metaspace2020",  # Only needed for experiment notebooks
          "psutil",  # Only needed for experiment notebooks
      ])
