from setuptools import setup, find_packages

setup(
    name='resmetric',
    version='1.0.0rc24',
    description='Explore and visualize resilience metrics and antifragility in performance graphs of self-adaptive '
                'systems.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    package_data={
            '': ['README.md'],
        },
    url='https://github.com/ferdinand-koenig/resmetric',
    author='Ferdinand Koenig',
    author_email='ferdinand@koenix.de',
    packages=find_packages(exclude=['development', '.idea', 'example', 'evaluation']),
    entry_points={
        'console_scripts': [
            'resmetric-cli=resmetric.cli:main',
        ],
    },
    install_requires=[
        'plotly>=5.23.0,<6.0.0',
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
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    python_requires='>=3.8, <4',
)
