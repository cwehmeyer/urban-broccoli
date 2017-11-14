from setuptools import setup
from setuptools.command.test import test as TestCommand
import sys

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]
    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ['urbanbroccoli']
    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)

description = '''
A Poisson solver written in Python for educational purposes.
'''

setup(
    cmdclass={'test': PyTest},
    name='urbanbroccoli',
    version='0.0.0',
    author='Christoph Wehmeyer',
    author_email='christoph.wehmeyer@fu-berlin.de',
    url='https://github.com/cwehmeyer/urban-broccoli',
    description=description,
    packages=['urbanbroccoli', 'urbanbroccoli.test'],
    setup_requires=['pytest-runner',],
    install_requires=['numpy'],
    tests_require=['pytest'],
    zip_safe=False)
