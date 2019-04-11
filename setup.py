from setuptools import setup, find_packages

long_description = '''
ISR (Image Super-Resolution) is a library to upscale and improve the quality of low resolution images.

Read the documentation at: https://idealo.github.io/image-super-resolution/

ISR is compatible with Python 3.6 and is distributed under the Apache 2.0 license.
'''

setup(
    name='ISR',
    version='2.0.5',
    author='Francesco Cardinale',
    author_email='testadicardi@gmail.com',
    description='Image Super Resolution',
    long_description=long_description,
    license='Apache 2.0',
    install_requires=['imageio', 'Keras==2.2.4', 'numpy==1.16.2', 'tensorflow==1.13.1', 'tqdm'],
    extras_require={
        'tests': ['pytest==4.3.0', 'pytest-cov==2.6.1'],
        'docs': ['mkdocs==1.0.4', 'mkdocs-material==4.0.2'],
        'gpu': ['tensorflow-gpu==1.13.1'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(exclude=('tests',)),
)
