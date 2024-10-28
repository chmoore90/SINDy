import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from pysindy.utils import linear_damped_SHO

import pysindy as ps

# ignore user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


np.random.seed(1000)  # Seed for reproducibility

# Integrator keywords for solve_ivp
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

# functions
a = 0.36
b = 100
c = 5**2

def linear_func(t, x, p=[a, c]):
    return [-p[0] * x[0] + p[1] * x[1], -p[1] * x[0] - p[0] * x[1]]

def vanderpol_func(t, x, p=[a, b, c]):
    return [x[1], x[1] * (p[0] - p[1] * x[0] ** 2) - p[2] * x[0]]

# Generate training data
function = vanderpol_func

dt = 0.01
t_train = np.arange(0, 25, dt)
t_train_span = (t_train[0], t_train[-1])
x0_train = [2, 0]
x_train = solve_ivp(function, t_train_span,
                    x0_train, t_eval=t_train, **integrator_keywords).y.T

# Fit the model
poly_order = 2
threshold = 0.05

model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=threshold),
    feature_library=ps.PolynomialLibrary(degree=poly_order),
)
model.fit(x_train, t=dt)
if function == linear_func:
    print("Linear model:")
elif function == vanderpol_func:
    print("Van der Pol model:")
model.print()


# Simulate and plot the results

x_sim = model.simulate(x0_train, t_train)
plot_kws = dict(linewidth=1)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(t_train, x_train[:, 0], "r", label="$x_0$", **plot_kws)
axs[0].plot(t_train, x_train[:, 1], "b", label="$x_1$", alpha=0.4, **plot_kws)
axs[0].plot(t_train, x_sim[:, 0], "k--", label="model", **plot_kws)
axs[0].plot(t_train, x_sim[:, 1], "k--")
axs[0].legend()
axs[0].set(xlabel="t", ylabel="$x_k$")

axs[1].plot(x_train[:, 0], x_train[:, 1], "r", label="$x_k$", **plot_kws)
axs[1].plot(x_sim[:, 0], x_sim[:, 1], "k--", label="model", **plot_kws)
axs[1].legend()
axs[1].set(xlabel="$x_1$", ylabel="$x_2$")
fig.show()
