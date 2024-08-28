from setuptools import setup, find_packages

setup(
    name='graph_clustering_framework',
    version='0.1',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'scikit-learn>=0.24.2',
        'python-igraph>=0.9.1',
        'autorank',
        'cairocffi',
        'IPython',
        'matplotlib',
        'networkx',
        'numpy',
        'pandas',
        'plotly>=5.23.0',
        'scipy>=1.6.0',
        'seaborn',
        'svgutils',
    ],
    extras_require={
        'dev': [
            'pytest',
            'sphinx',
        ],
    },
    entry_points={
        'console_scripts': [
            'example_script=src.example:main',
        ],
    },
    author='Seu Nome',
    author_email='aartur.porto@gmail.com',
    description='A framework for graph construction and data clustering',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Artur-Porto/graph-clustering-framework',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
