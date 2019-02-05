from setuptools import setup, find_packages

with open('README.md') as file:
    readme = file.read()

setup(name="realML",
      version="2.5.0",
      description="ICSI provided machine learning primitives for DARPA D3M project",
      packages=find_packages(),
      entry_points={
          'd3m.primitives' : [
              'classification.logistic.TensorMachinesBinaryClassification = realML.kernel:TensorMachinesBinaryClassification',
              'regression.polynomial.TensorMachinesRegularizedLeastSquares = realML.kernel:TensorMachinesRegularizedLeastSquares',
              'regression.gaussiankernel.RFMPreconditionedGaussianKRR = realML.kernel:RFMPreconditionedGaussianKRR',
              'regression.polynomialkernel.RFMPreconditionedPolynomialKRR = realML.kernel:RFMPreconditionedPolynomialKRR',
              'regression.l1norm.FastLAD = realML.matrix:FastLAD',
              'feature_extraction.matrixfactorization.L1LowRank = realML.matrix:L1LowRank'
          ],
      },
      url="https://github.com/ICSI-RealML/realML",
      long_description=readme,
      author="International Computer Science Institute",
      author_email="gittea@rpi.edu",
      license="GPL",
      keywords=['d3m_primitive', 'machine learning', 'regression', 'dimensionality reduction', 'low rank factorization', 'featurization', 'sufficient dimensionality reduction', 'kernel methods'],
      install_requires=[
          'd3m',
          'numpy',
          'scipy',
          'scikit-learn'
      ],
      zip_safe=False
)
