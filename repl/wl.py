from nvml._study.wls import fe1, ff  # noqa
from nvml.wl.classify import SilhouettePlot  # noqa
from nvml.wl.kernel.kern_study import make_kernel_and_plot, plot_for_all_schemes  # noqa
import matplotlib.pyplot as plt  # noqa
import iplotx as ipx  # noqa
from grakel import VertexHistogram, WeisfeilerLehman, ShortestPath  # noqa

import matplotlib

matplotlib.use("module://matplotlib-backend-kitty")

# kern, model, graphs = fc()
# graphs, kern = fe()
# fgraphs = ff()
gkdm = fe1()
