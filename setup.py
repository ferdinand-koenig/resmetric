from setuptools import setup, find_packages

setup(
    name='resmetric',
    version='1.0.0-rc.3',
    description='A Python module for enhancing Plotly figures with resilience-related metrics. (Anonymized version)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['development', '.idea', 'example', 'evaluation']),
    entry_points={
        'console_scripts': [
            'resmetric-cli=resmetric.cli:main',
        ],
    },
    install_requires=[
        'plotly>=5.23.0',
        'numpy>=1.24.4',
        'scipy>=1.10.1',
        'scikit-optimize>=0.10.2',
        'pwlf>=2.2.1'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8, <4',
)
