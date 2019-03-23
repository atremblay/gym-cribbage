# -*- coding: utf-8 -*-
# @Author: Marc-Antoine
# @Date:   2019-03-17 17:18:42
# @Last Modified by:   Marc-Antoine Belanger
# @Last Modified time: 2019-03-17 17:20:31

from setuptools import setup, find_namespace_packages

setup(
    name='gym_cribbage',
    version='0.0.1',
    install_requires=['gym', 'numpy'],  # And any other dependencies cribbage needs
    packages=find_namespace_packages(where='src'),
    package_dir={'': 'src'}
)
