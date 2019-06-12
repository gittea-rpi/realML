from setuptools import setup, find_packages

with open('README.md') as file:
    readme = file.read()

setup(name="realML",
      version="2.7.4",
      description="ICSI provided machine learning primitives for DARPA D3M project, focusing on fast kernel methods and matrix factorizations ",
      packages=find_packages(),
      entry_points={
          'd3m.primitives' : [
              #'classification.tensor_machines_binary_classification.TensorMachinesBinaryClassification = realML.kernel:TensorMachinesBinaryClassification',
              'regression.tensor_machines_regularized_least_squares.TensorMachinesRegularizedLeastSquares = realML.kernel:TensorMachinesRegularizedLeastSquares',
              'regression.rfm_precondition_ed_gaussian_krr.RFMPreconditionedGaussianKRR = realML.kernel:RFMPreconditionedGaussianKRR',
              'regression.rfm_precondition_ed_polynomial_krr.RFMPreconditionedPolynomialKRR = realML.kernel:RFMPreconditionedPolynomialKRR',
              'regression.fast_lad.FastLAD = realML.matrix:FastLAD',
              #'feature_extraction.l1_low_rank.L1LowRank = realML.matrix:L1LowRank',
              #'feature_extraction.sparse_pca.sparsepca = realML.matrix:sparse_pca',
          ],
      },
      url="https://github.com/ICSI-RealML/realML",
      long_description=readme,
      author="International Computer Science Institute",
      author_email="gittea@rpi.edu",
      license="GPL",
      keywords=['d3m_primitive', 'machine learning', 'regression', 'dimensionality reduction', 'low rank factorization', 'featurization', 'sufficient dimensionality reduction', 'kernel methods'],
      install_requires=[
          'regex==2017.4.5',
          'numpy>=1.14.0',
          'scipy<=1.2.1,>=0.19.0',
          'scikit-learn>=0.18.1'
      ],
      zip_safe=False
)
