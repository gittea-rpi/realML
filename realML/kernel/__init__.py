#from .rfmpreconditionedgaussiankrr import RFMPreconditionedGaussianKRR
#from .rfmpreconditionedpolynomialkrr import RFMPreconditionedPolynomialKRR
from pkg_resources import get_distribution
__author__ = 'ICSI'
__version__ = get_distribution('realML').version
from .TensorMachinesBinaryClassification import TensorMachinesBinaryClassification
