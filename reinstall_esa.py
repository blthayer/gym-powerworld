"""Helper to reinstall ESA package from source. It is assumed
that the repo exists at the same directory level as pw_learn.

https://pip.pypa.io/en/latest/user_guide/#using-pip-from-your-program
"""
import subprocess
import sys
import os

# Uninstall.
subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'esa'])

# Move into the ESA repo.
os.chdir('../esa')

# Install.
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '.'])
