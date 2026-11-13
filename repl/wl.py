from nvml._study.wls import fc, fe  # noqa
from nvml.wl.classify import SilhouettePlot  # noqa
import matplotlib.pyplot as plt  # noqa
import iplotx as ipx  # noqa
from grakel import VertexHistogram, WeisfeilerLehman, ShortestPath  # noqa

import matplotlib

matplotlib.use("module://matplotlib-backend-kitty")

# kern, model, graphs = fc()
graphs = fe()
