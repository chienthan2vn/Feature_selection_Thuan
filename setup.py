from setuptools import setup, find_packages

from pso_lr import __version__

setup(
    name='pso_lr',
    version=__version__,
    description='Feature Selection using PSO_LR.',
    url='https://github.com/chienthan2vn/Feature_selection_Thuan',
    author='Thuan Luong',
    author_email='thuanluong19102001@gmail.com',
    packages=find_packages(),
)