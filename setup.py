from setuptools import setup, find_packages

setup(
    name='graph_clustering_framework',
    version='0.1',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'networkx',
        'scikit-learn',
        'python-igraph',
        'cairocffi',  # substitui python-cairo para compatibilidade com pip
        'svgutils',
        'autorank',
        'pandas',
        'mpl_toolkits.mplot3d',  # mpl_toolkits é uma parte do matplotlib
        'IPython',
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

