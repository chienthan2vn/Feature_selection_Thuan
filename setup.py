from setuptools import setup, find_packages

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setup(
    name='Feature_selection_Thuan',
    version='0.0.2',
    description='Feature Selection',
    url='https://github.com/chienthan2vn/Feature_selection_Thuan',
    author='Thuan Luong',
    author_email='thuanluong19102001@gmail.com',
    long_description = long_description,
    long_description_content_type = "text/markdown",
    project_urls = {
        "Bug Tracker": "package issues URL",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "src"},
    packages = find_packages(where="src"),
    python_requires = ">=3.6"
)