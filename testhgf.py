import numpy as np
import hgf


p2 = hgf.StateNodeParameters(1, np.inf, 0.0, [], -2.0, [])
p1 = hgf.StateNodeParameters(0, 1, 0.0, [], -12.0, [1])
pU = hgf.InputNodeParameters(0.0, None)

x2 = hgf.StateNode(p2, [], [])
x1 = hgf.StateNode(p1, [], [x2])
xU = hgf.InputNode(pU, x1, None)

xU.input(0.5)
