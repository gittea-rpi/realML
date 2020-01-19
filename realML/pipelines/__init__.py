from .RFMPreconditionedGaussianKRR_pipeline import RFMPreconditionedGaussianKRRPipeline
from .RFMPreconditionedGaussianKRR_pipeline_196_autoMpg import RFMPreconditionedGaussianKRRPipeline_196_autoMpg
#from .RFMPreconditionedGaussianKRR_pipeline_26_radon_seed import RFMPreconditionedGaussianKRRPipeline_26_radon_seed

from .RFMPreconditionedPolynomialKRR_pipeline import RFMPreconditionedPolynomialKRRPipeline

from .TensorMachinesRegularizedLeastSquares_pipeline import TensorMachinesRegularizedLeastSquaresPipeline

from .TensorMachinesBinaryClassification_pipeline import TensorMachinesBinaryClassificationPipeline

#from .L1LowRank_pipeline import L1LowRankPipeline

#from .FastLAD_pipeline import FastLADPipeline



#from .sparsepca_pipeline import sparsepcaPipeline
from .sparsepca_pipeline_2 import sparsepcaPipeline2
from .sparsepca_pipeline_3 import sparsepcaPipeline3
from .sparsepca_pipeline_4 import sparsepcaPipeline4
from .sparsepca_pipeline_5 import sparsepcaPipeline5
#from .sparsepca_pipeline_6 import sparsepcaPipeline6

from .robustsparsepca_pipeline import robustsparsepcaPipeline
from .robustsparsepca_pipeline_2 import robustsparsepcaPipeline2

from .randomizedpolypca_pipeline import randomizedpolypcaPipeline
#
#from .randomizedpolypca_pipeline_2 import randomizedpolypcaPipeline2
from .randomizedpolypca_pipeline_3 import randomizedpolypcaPipeline3


__all__ = ["RFMPreconditionedGaussianKRRPipeline", 
           "RFMPreconditionedGaussianKRRPipeline_196_autoMpg", 
           #"RFMPreconditionedGaussianKRRPipeline_26_radon_seed", 
           "RFMPreconditionedPolynomialKRRPipeline",
           "TensorMachinesRegularizedLeastSquaresPipeline",
           #"TensorMachinesBinaryClassificationPipeline",
           #"L1LowRankPipeline",
           #"FastLADPipeline",
           #"sparsepcaPipeline",
           "sparsepcaPipeline2",
           "sparsepcaPipeline3",
           "sparsepcaPipeline4",
           "sparsepcaPipeline5",
           #"sparsepcaPipeline6",           
           "robustsparsepcaPipeline",
           "robustsparsepcaPipeline2",
           "randomizedpolypcaPipeline",
           #"randomizedpolypcaPipeline2"
           "randomizedpolypcaPipeline3"
           ]
