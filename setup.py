from setuptools import setup, find_packages

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setup(
    name='machinegnostics',
    version='0.0.0',
    packages=find_packages(),
    install_requires=[],
    author='Nirmal Parmar',
    author_email='machinegnostics@gmail.com',
    description='Machine Gnostics is an open-source initiative that seeks to redefine the mathematical underpinnings of machine learning. While most conventional ML libraries are grounded in probabilistic and statistical frameworks, Machine Gnostics explores alternative paradigms—drawing from deterministic algebra, information theory, and geometric methods. This approach opens new avenues for building robust, interpretable, and reliable analysis tools that can withstand the limitations of traditional models.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MachineGnostics/machinegnostics',
    classifiers=[
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'seaborn',
        'joblib',
    ],
    )
