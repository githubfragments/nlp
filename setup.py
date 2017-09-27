from setuptools import setup
from setuptools import find_packages


setup(name='nlp',
      version='1.0',
      description='utils for nlp',
      author='David Vaughn',
#       author_email='francois.chollet@gmail.com',
#       url='https://github.com/fchollet/keras',
#       download_url='https://github.com/fchollet/keras/tarball/2.0.3',
#       license='MIT',
#       install_requires=['theano', 'pyyaml', 'six'],
#       extras_require={
#           'h5py': ['h5py'],
#           'visualize': ['pydot-ng'],
#           'tests': ['pytest',
#                     'pytest-pep8',
#                     'pytest-xdist',
#                     'pytest-cov'],
#       },
      packages=find_packages()
      )