from setuptools import setup, find_packages

setup(name="realML",
      version="0.2",
      description="ICSI provided machine learning primitives for DARPA D3M project",
      entry_points={
          'd3m.primitives' : [
              'realML.kernel.TensorMachinesBinaryClassification = realML.kernel:TensorMachinesBinaryClassification',
              'realML.kernel.TensorMachinesRegularizedLeastSquares = realML.kernel:TensorMachinesRegularizedLeastSquares'
              'realML.kernel.RFMPreconditionedGaussianKRR = realML.kernel:RFMPreconditionedGaussianKRR'
              'realML.kernel.RFMPreconditionedPolynomialKRR = realML.kernel:RFMPreconditionedPolynomialKRR'
          ],
      },
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
      keywords=['d3m_primitive', 'machine learning', 'regression', 'dimensionality reduction', 'low rank factorization', 'featurization', 'sufficient dimensionality reduction', 'kernel methods'],
      install_requires=[
          'numpy>=1.14.0',
          'scipy>=0.13.3',
          'scikit-learn>=0.18.1'
      ],
      packages=find_packages(),
      zip_safe=False
)
