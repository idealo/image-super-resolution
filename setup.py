from setuptools import setup, find_packages

long_description = '''
ISR (Image Super-Resolution) is a library to upscale and improve the quality of low resolution images.

It includes the Keras implementations of:

- The super-scaling Residual Dense Network described in Residual Dense Network
  for Image Super-Resolution (Zhang et al. 2018)
- The super-scaling Residual in Residual Dense Network described
  in ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks (Wang et al. 2018)
- A multi-output version of the Keras VGG19 network for deep features extraction used in the perceptual loss
- A custom discriminator network based on the one described in
  Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (SRGANS, Ledig et al. 2017)

Read the documentation at: https://idealo.github.io/image-super-resolution/

ISR is compatible with Python 3.6 and is distributed under the Apache 2.0 license.
'''

setup(
    name='ISR',
    version='1.9.2',
    author='Francesco Cardinale',
    author_email='testadicardi@gmail.com',
    description='Image Super Resolution',
    long_description=long_description,
    license='Apache 2.0',
    install_requires=['imageio', 'Keras==2.2.4', 'numpy==1.16.2', 'tensorflow==1.13.1', 'tqdm'],
    extras_require={
        'tests': ['pytest==4.3.0', 'pytest-cov==2.6.1'],
        'docs': ['mkdocs==1.0.4', 'mkdocs-material==4.0.2'],
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
