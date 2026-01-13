from setuptools import setup, find_packages
import os

# Utility function to read the README file.
# This makes the project description on PyPI look nice if you release it.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="lit_gpt",
    version="0.1.0",
    description="A customized implementation of Lit-GPT",
    # Reads the long description from your README.md
    long_description=read('README.md') if os.path.exists('README.md') else "",
    long_description_content_type='text/markdown',
    
    # author="Your Name",
    # author_email="your.email@example.com",
    # url="https://github.com/yourusername/lit_gpt",
    
    # find_packages() automatically discovers directories containing __init__.py
    packages=find_packages(),
    
    # DEPENDENCIES: List the external libraries your module needs here.
    # Since this is lit_gpt, you likely need torch and lightning.
    install_requires=[
        # "torch>=2.0.0",
        # "lightning>=2.0.0",
        # "numpy",
        # "tokenizers",
        # Add other dependencies here
    ],
    
    # Python version requirement
    # python_requires='>=3.8',
    
    # OPTIONAL: Classifiers help users find your project by category
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: Apache Software License",
    #     "Operating System :: OS Independent",
    # ],
)