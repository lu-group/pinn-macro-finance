import os

os.environ["DDEBACKEND"] = "tensorflow"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from math import pi

import deepxde as dde
from deepxde.backend import tf

dde.config.disable_xla_jit()

k = 0.005
a_train = dde.Variable(2.0)  # a = a_train / 10
phi = 10
w = 1
r = 0.03
rl_train = dde.Variable(0.0)  # rl = (sigmoid(rl_train) * 2 + 3.5) / 100
sigma = 0.08
cf = 0.03
m_bar = 0.1
ce = 0.1
beta_M = 1e3
beta = 0.05
epsilon = 1
D0 = 1.0
eps = 1

# 0.0075 is a small value chosen so the v > e region is continuous
ind_g_eps = 0.0075

z_target = 7.515612486080857
l_star_target = 0.006457735763252105

z_min = 0.2
z_max = 10
z_mean = 5
e_min = 0.01
e_max = 1.2

stdev = (z_max - z_min) / 4
area = (z_max - z_min) * (e_max - e_min)


def f(m, n):
    a = a_train / 10
    return m * n**a


def psi(x):
    e = x[:, 0]
    z = x[:, 1]

    psi_unnorm = (
        1 / (stdev * tf.sqrt(2 * pi)) * tf.exp(-0.5 * ((z - z_mean) / stdev) ** 2)
    )
    psi = psi_unnorm * 7.471114 * tf.cast(e <= 0.15, dtype=tf.float32)

    return psi


def integrate(x):
    # Computes double integral over the entire region
    return tf.reduce_sum(x) / tf.cast(tf.size(x), dtype=tf.float32) * area


def pde(x, y):
    e = x[:, 0:1]
    z = x[:, 1:2]
    dv_e = dde.grad.jacobian(y, x, i=0, j=0)
    dv_z = dde.grad.jacobian(y, x, i=0, j=1)
    dv_zz = dde.grad.hessian(y, x, component=0, i=1, j=1)
    dg_e = dde.grad.jacobian(y, x, i=1, j=0)
    dg_z = dde.grad.jacobian(y, x, i=1, j=1)
    dg_zz = dde.grad.hessian(y, x, component=1, i=1, j=1)

    v = y[:, 0:1]
    g = y[:, 1:2]
    mu = -0.005 * (z - 5)

    a = a_train / 10
    rl = (tf.math.sigmoid(rl_train) * 2 + 3.5) / 100

    ind = tf.cast(
        ((rl - r) * z * a / w) ** (1 / (1 - a)) < (phi * e / z) ** (1 / a),
        dtype=tf.float32,
    )
    l_star = tf.minimum(
        ((rl - r) * z * a / w) ** (1 / (1 - a)), (phi * e / z) ** (1 / a)
    )
    pi_star = 2 * (rl - r) * z * l_star**a + e * r - 2 * w * l_star - cf
    zeta = k * tf.maximum(tf.cast(0, dtype=tf.float32), phi * e - f(z, l_star))
    v_u = e

    hjb = r * v - tf.maximum(
        pi_star * (1 + dv_e)
        + (1 - dv_e) * ind * zeta
        + dv_z * mu
        + 1 / 2 * dv_zz * sigma**2,
        r * v_u,
    )

    psi_val = psi(x)
    psi_val = tf.reshape(psi_val, [-1, 1])

    m = 0.1

    mu_z = -0.005
    mu_e = pi_star - ind * zeta
    l_star_e = (1 - ind) * (phi / z) ** (1 / a) * (1 / a) * (e ** ((1 / a) - 1))
    pi_star_e = (
        2 * (rl - r) * z * a * (l_star) ** (a - 1) * l_star_e + r - 2 * w * l_star_e
    )
    zeta_e = (
        k
        * tf.cast(phi * e > z * l_star**a, dtype=tf.float32)
        * (phi - z * a * l_star ** (a - 1) * l_star_e)
    )
    mu_ee = pi_star_e - zeta_e * ind

    ind_g = tf.cast((v - e) > ind_g_eps, dtype=tf.float32)
    kfe = (
        -mu_z * g
        - mu * dg_z
        - mu_ee * g
        - mu_e * dg_e
        + dg_zz * sigma**2 / 2
        + m * psi_val * ind_g
    ) * ind_g

    # No error on first iteration
    if tf.reduce_sum(g) == 0:
        z_target_pred = 0.0
        l_star_target_pred = 0.0
    else:
        gp = g / integrate(g)
        z_target_pred = integrate(z * gp)
        l_star_target_pred = integrate(l_star * gp)

    z_target_loss = (z_target_pred - z_target) * tf.ones(tf.shape(x))
    l_star_target_loss = (l_star_target_pred - l_star_target) * tf.ones(tf.shape(x))

    return [hjb, kfe, z_target_loss, l_star_target_loss]


geom = dde.geometry.Rectangle([e_min, z_min], [e_max, z_max])


def boundary_z(x, on_boundary):
    return on_boundary and (np.isclose(x[1], z_min) or np.isclose(x[1], z_max))


def boundary_e(x, on_boundary):
    return on_boundary and np.isclose(x[0], e_min)


def output_transform(x, y):
    e, z = x[:, 0:1], x[:, 1:2]
    v, g = y[:, 0:1], y[:, 1:2]

    ind_g = tf.cast(v > e + ind_g_eps, dtype=tf.float32)

    return tf.concat([v, (g * ind_g) ** 2], axis=1)


def func_bc(x, y):
    z = x[:, 1:2]
    g = y[:, 1:2]

    mu = -0.005 * (z - 5)
    dg_z = dde.grad.jacobian(y, x, i=1, j=1)

    return -mu * g + 0.5 * sigma**2 * dg_z


bcD_v = dde.DirichletBC(geom, lambda x: 0.01, boundary_e, component=0)
bcN_v = dde.NeumannBC(geom, lambda x: 0, boundary_z, component=0)
bcN_g = dde.icbc.OperatorBC(
    geom,
    lambda x, y, _: func_bc(x, y),
    boundary_z,
)

e_true = np.loadtxt(open("../data/hjb/hjb_e.csv", "r"), delimiter=",", dtype=np.float32).flatten()
z_true = np.loadtxt(open("../data/hjb/hjb_z.csv", "r"), delimiter=",", dtype=np.float32).flatten()
v_true = np.loadtxt(open("../data/hjb/hjb_v.csv", "r"), delimiter=",", dtype=np.float32).flatten()[
    :, None
]
g_true = np.loadtxt(open("../data/hjb/hjb_g.csv", "r"), delimiter=",", dtype=np.float32).flatten()[
    :, None
]

X = np.vstack((e_true, z_true)).T


def func(x):
    return np.hstack((griddata(X, v_true, x), griddata(X, g_true, x)))


def v_l2(y_true, y_pred):
    v_true = y_true[:, 0]
    v_pred = y_pred[:, 0]
    return np.linalg.norm(v_true - v_pred) / np.linalg.norm(v_true)


def g_l2(y_true, y_pred):
    g_true = y_true[:, 1]
    g_pred = y_pred[:, 1]
    g_true = np.maximum(g_true, 0)
    g_pred = np.maximum(g_pred, 0)
    g_true = g_true / np.sum(g_true)
    g_pred = g_pred / np.sum(g_pred)
    return np.linalg.norm(g_true - g_pred) / np.linalg.norm(g_true)


data = dde.data.PDE(
    geom,
    pde,
    [bcD_v, bcN_v, bcN_g],
    num_domain=16384,
    num_boundary=512,
    num_test=16384,
    solution=func,
)

net = dde.nn.FNN([2] + [64] * 6 + [2], "tanh", "Glorot uniform")
net.apply_output_transform(output_transform)
model = dde.Model(data, net)

lw = [1e6, 5e4, 1e1, 1e6, 1e2, 1e3, 1e5]
variable = dde.callbacks.VariableValue([rl_train, a_train], period=1000, precision=9)

model.compile(
    "adam",
    lr=1e-3,
    decay=("inverse time", 2500, 1.0),
    loss_weights=lw,
    external_trainable_variables=[rl_train, a_train],
    metrics=[v_l2, g_l2],
)
losshistory, train_state = model.train(
    epochs=150000, callbacks=[variable], display_every=1000
)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

N = 400  # Create N x N heatmaps of the functions
e_vals = np.linspace(e_min, e_max, N)
z_vals = np.linspace(z_min, z_max, N)

v_pred = np.empty((N, N))
g_pred = np.empty((N, N))
for i in range(N):
    for j in range(N):
        model_pred = model.predict([[e_vals[j], z_vals[i]]])
        v_pred[i, j] = model_pred[0, 0]
        g_pred[i, j] = model_pred[0, 1]

e_true = np.genfromtxt("../data/hjb/hjb_e.csv", delimiter=",").T
z_true = np.genfromtxt("../data/hjb/hjb_z.csv", delimiter=",").T
v_true = np.genfromtxt("../data/hjb/hjb_v.csv", delimiter=",").T
g_true = np.maximum(np.genfromtxt("../data/hjb/hjb_g.csv", delimiter=",").T, 0)


def integrate_np(f):
    return np.sum(f) / np.size(f) * area


# Print the results for the predicted variables
rl_true = 0.043343
rl_pred = (1 / (1 + np.exp(-rl_train)) * 2 + 3.5) / 100

a_true = 0.3
a_pred = a_train / 10

ce_true = 0.1
# Solve for c_e using the technique in Section
e_vals_reshaped = np.reshape(e_vals, (1, N))
z_vals_reshaped = np.reshape(z_vals, (N, 1))
psi_unnorm = (
    1
    / (stdev * np.sqrt(2 * pi))
    * np.exp(-0.5 * ((z_vals_reshaped - z_mean) / stdev) ** 2)
)
psi = psi_unnorm * 7.471114 * (e_vals_reshaped <= 0.15)
ce_pred = integrate_np(v_pred * psi)

# Calculate the true value of g using Section 3.2.1
l_star_pred = np.minimum(
    ((rl_pred - r) * z_true * a_pred / w) ** (1 / (1 - a_pred)),
    (phi * e_true / z_true) ** (1 / a_pred),
)
L = integrate_np(z_vals * l_star_pred**a_pred * g_pred)
m_pred = (beta / (rl_pred - r) - D0) / L * 0.1
g_pred = g_pred * (m_pred / 0.1)

# Plot predicted and true values of v and g
def plot_heatmap(fun, title):
    fig, ax = plt.subplots()
    c = ax.pcolormesh(e_vals, z_vals, fun, cmap="rainbow")
    ax.set_aspect(0.1)
    ax.set_ylabel("z")
    ax.set_xlabel("e")
    ax.set_title(title)
    cbar = fig.colorbar(c, ax=ax)

    plt.show()


plot_heatmap(v_pred, "PINN")
plot_heatmap(v_true, "Reference")
plot_heatmap(g_pred, "PINN")
plot_heatmap(g_true, "Reference")

# Print values of parameters and endogenous variable r^l
print("true r^l value: %.4f\npred r^l value: %.4f\n" % (rl_true, rl_pred))
print("true alpha value: %.4f\npred alpha value: %.4f\n" % (a_true, a_pred))
print("true c_e value: %.4f\npred c_e value: %.4f" % (ce_true, ce_pred))


# Calculate true and predicted cumulative g
gp_pred = g_pred / integrate_np(g_pred)
gp_true = g_true / integrate_np(g_true)

gc_true = np.empty((400, 400), dtype=float)
gc_pred = np.empty((400, 400), dtype=float)

for i in range(400):
    for j in range(400):
        # Val refers to gc[i, j]
        if i == 0 or j == 0:
            val_pred = 0
            val_true = 0
        else:
            gp_true_section = gp_true[:i, :j]
            gp_pred_section = gp_pred[:i, :j]

            val_true = integrate_np(gp_true_section) * (i / 400) * (j / 400)
            val_pred = integrate_np(gp_pred_section) * (i / 400) * (j / 400)

        gc_pred[i, j] = val_pred
        gc_true[i, j] = val_true

# Calculate the L2 relative errors of v and cumulative g
def l2(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b) * 100


v_l2_err = l2(v_pred, v_true)
gc_l2_err = l2(gc_pred, gc_true)

print("v L2 relative error: %.2f" % v_l2_err)
print("g_c L2 relative error: %.2f" % gc_l2_err)
