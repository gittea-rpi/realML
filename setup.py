from setuptools import setup, find_packages

with open('README.md') as file:
    readme = file.read()

setup(name="realML",
      version="2.5.0",
      description="ICSI provided machine learning primitives for DARPA D3M project",
      packages=find_packages(),
      entry_points={
          'd3m.primitives' : [
              'realML.TensorMachinesBinaryClassification = realML.kernel:TensorMachinesBinaryClassification',
              'realML.TensorMachinesRegularizedLeastSquares = realML.kernel:TensorMachinesRegularizedLeastSquares',
              'realML.RFMPreconditionedGaussianKRR = realML.kernel:RFMPreconditionedGaussianKRR',
              'realML.RFMPreconditionedPolynomialKRR = realML.kernel:RFMPreconditionedPolynomialKRR',
              'realML.FastLAD = realML.matrix:FastLAD',
              'realML.L1LowRank = realML.matrix:L1LowRank'
          ],
      },
      url="https://github.com/ICSI-RealML/realML",
      long_description=readme,
      author="International Computer Science Institute",
      author_email="gittea@rpi.edu",
      license="GPL",
      keywords=['d3m_primitive', 'machine learning', 'regression', 'dimensionality reduction', 'low rank factorization', 'featurization', 'sufficient dimensionality reduction', 'kernel methods'],
      install_requires=[
          'numpy>=1.14.0',
          'scipy>=0.13.3',
          'scikit-learn>=0.18.1'
      ],
      zip_safe=False
)
