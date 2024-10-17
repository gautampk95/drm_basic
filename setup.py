#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# setup.py file
from setuptools import setup, find_packages


# In[ ]:


setup(
    name='drm_basic',
    version='0.1',
    author='Gautham Prasad K',
    author_email='gautamprasadk95@gmail.com',
    description='A library for building, summarizing, and fitting basic dose-response models for continuous response data (support for binomial, poisson and event data is to be added in the future)',
    packages=find_packages(),
    install_requires=[
        'numpy',  # List of dependencies
        'pandas',
        'scipy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)


# In[ ]:




