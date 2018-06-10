from setuptools import setup, find_packages

with open('README.md') as file:
    readme = file.read()

setup(name="realML",
      version="2.0.0",
      description="ICSI provided machine learning primitives for DARPA D3M project",
      packages=find_packages(),
      entry_points={
          'd3m.primitives' : [
              'realML.kernel.TensorMachinesBinaryClassification = realML.kernel:TensorMachinesBinaryClassification',
              'realML.kernel.TensorMachinesRegularizedLeastSquares = realML.kernel:TensorMachinesRegularizedLeastSquares',
              'realML.kernel.RFMPreconditionedGaussianKRR = realML.kernel:RFMPreconditionedGaussianKRR',
              'realML.kernel.RFMPreconditionedPolynomialKRR = realML.kernel:RFMPreconditionedPolynomialKRR'
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
