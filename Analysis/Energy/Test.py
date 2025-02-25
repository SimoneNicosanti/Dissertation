import numpy as np
import pyRAPL

pyRAPL.setup()


@pyRAPL.measureit(number=5_000_000)
def foo(a, b, c):
    a * b
    # Instructions to be evaluated.


val_1 = np.random.random()
val_2 = np.random.random()
val_3 = np.random.random()

foo(val_1, val_2, val_3)
