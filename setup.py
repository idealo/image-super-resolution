from setuptools import setup, find_packages

setup(
    name='ISR',
    version='1.9.0',
    author='Francesco Cardinale',
    author_email='testadicardi@gmail.com',
    description='Image Super Resolution',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    license='Apache 2.0',
    install_requires=['imageio', 'Keras==2.2.4', 'numpy==1.16.2', 'tensorflow==1.13.1', 'tqdm'],
    extras_require={'tests': ['pytest==4.3.0', 'pytest-cov==2.6.1']},
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
