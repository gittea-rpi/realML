from setuptools import setup, find_packages

setup(name="realML",
      version="0.1.0",
      description="ICSI provided machine learning primitives for DARPA D3M project",
      url="http://github.com/alexgittens/realML",
      author="Alex Gittens",
      author_email="gittea@rpi.edu",
      license="GPL",
      classifiers=[
                   'Development Status :: 3 - Alpha',
                   'License :: OSI Approved :: GNU General Public License (GPL)',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.3',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   ],
      keywords=['machine learning', 'regression', 'dimensionality reduction', 'low rank factorization'],
      install_requires=[
          'numpy>=1.12.0',
          'scipy>=0.18.1',
          'scikit-learn>=0.18.1'
      ],
      packages=find_packages(),
      zip_safe=False
)
