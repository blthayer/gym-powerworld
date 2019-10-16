"""
Reference:
https://github.com/pypa/sampleproject/blob/master/setup.py
"""
from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file

with open(path.join(here, 'README.md'), encoding='utf-8') as f:

    long_description = f.read()

setup(name='gym_powerworld',
      version='0.0.1',
      description=('OpenAI gym environment for interfacing with PowerWorld '
                   'Simulator'),
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/blthayer/gym-powerworld',
      author='Brandon Thayer',
      author_email='blthayer@tamu.edu',
      # For a list of valid classifiers, see
      # https://pypi.org/classifiers/
      classifiers=[
            'Development Status :: 1 - Planning',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: Python :: 3.6'
            # TODO: Add 3.7 after figuring out pywin32 issues
      ],
      keywords=('deep reinforcement learning machine PowerWorld smart grid '
                'voltage control electric power system'),
      # TODO: Add 3.7 after figuring out pywin32 issues
      python_requires='==3.6.*',
      install_requires=[
            'gym',
            'numpy',
            'pandas'
      ]
)