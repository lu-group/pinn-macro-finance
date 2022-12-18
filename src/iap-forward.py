import os

os.environ["DDEBACKEND"] = "tensorflow.compat.v1"
import deepxde as dde
import numpy as np
from deepxde.backend import tf
from deepxde.callbacks import Callback
from deepxde.callbacks import OperatorPredictor
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import bisect

import torch

dde.config.set_default_float("float64")

a = 0.11
a_ = 0.05
rho = 0.06
r = 0.05
sigma = 0.025
delta = 0.03
delta_ = 0.08
kappa = 10
eta_s_var = dde.Variable(0.4, dtype=tf.float64)
eta_s_real = 0.364762616462568
q_ = 0.48616429
C = dde.Variable(1.0, dtype=tf.float64)

lim = 0


def save_data():
    eta_s_var = variable.get_value()[1]
    eta_s = eta_s_var
    t = np.linspace(lim, eta_s, 10000)
    t = t.reshape(10000, 1)

    sol_pred = model.predict(t)
    q_pred = sol_pred[:, 0:1]
    theta_pred = 1 / sol_pred[:, 1:2]

    np.savetxt("eta_%d.txt", t)
    np.savetxt("q_pred_%d.txt", q_pred)
    np.savetxt("theta_pred_%d.txt", theta_pred)


class Update(OperatorPredictor):
    def __init__(self, x, eta_s_var, component_x=0, component_y=0):
        def info(x, y):
            return (
                dde.grad.jacobian(y, x, i=0, j=component_x),
                dde.grad.jacobian(y, x, i=1, j=component_x),
                eta_s_var,
            )

        super().__init__(x, info)

    def on_batch_begin(self):
        pts = np.linspace(lim, 1, 10000)
        pts = np.reshape(pts, (-1, 1))
        y = self.model._outputs(False, pts)
        q = y[:, 0]
        theta_ = y[:, 1]
        tf_qp, tf_theta_p, tf_eta_s_var = self.op(
            self.model.net.inputs, self.model.net.outputs
        )
        qp = self.model.sess.run(tf_qp, feed_dict=self.model.net.feed_dict(False, pts))
        theta_p = self.model.sess.run(
            tf_theta_p, feed_dict=self.model.net.feed_dict(False, pts)
        )
        eta = pts

        q = np.reshape(q, np.shape(qp))
        theta_ = np.reshape(theta_, np.shape(theta_p))

        eta_s_var = self.model.sess.run(
            tf_eta_s_var, feed_dict=self.model.net.feed_dict(False, pts)
        )
        eta_s = eta_s_var
        ind = eta < eta_s

        epoch_num = self.model.train_state.epoch
        # Plot q, theta_ every 10k epochs (for testing)
        if epoch_num % 10000 == 0:
            plt.plot(eta, q, label="q")
            plt.legend()
            plt.xlim(lim, eta_s)
            plt.ylim(0, 2)
            plt.show()

            plt.plot(eta, theta_, label="theta_")
            plt.xlim(lim, eta_s)
            plt.ylim(0, 1.5)
            plt.legend()
            plt.show()
        if epoch_num % 1000 == 0 and epoch_num >= 1000:
            # q, theta_ increasing -> bisect works
            qp_sum = np.sum(np.minimum(0, qp * ind))
            theta_p_sum = np.sum(np.minimum(0, theta_p * ind))
            if np.isclose(qp_sum, 0) and np.isclose(theta_p_sum, 0):
                cf1 = 1 / (theta_ + 1e-6) * (qp**2) * (a - a_ + q * (delta_ - delta))
                cf2 = -2 / (theta_ + 1e-6) * qp * (q + eta * qp) * (
                    a - a_ + q * (delta_ - delta)
                ) + sigma**2 * q**3 * (-theta_p) / ((theta_ + 1e-6) ** 2)
                cf3 = (
                    1
                    / (theta_ + 1e-6)
                    * (2 * q * qp * eta + (eta * qp) ** 2 + q**2)
                    * (a - a_ + q * (delta_ - delta))
                    - sigma**2 * q**3 * (-theta_p) / ((theta_ + 1e-6) ** 2) * eta
                )

                disc = cf2**2 - 4 * cf1 * cf3
                rdisc = tf.cast(disc > 0, tf.float32) * disc

                res1 = (-cf2 + rdisc**0.5) / (2 * cf1 + 1e-6)
                res2 = (-cf2 - rdisc**0.5) / (2 * cf1 + 1e-6)
                cast1 = tf.cast(res1 < eta + q / qp, tf.float32) * tf.cast(
                    res1 > eta, tf.float32
                )
                cast2 = tf.cast(res2 < eta + q / qp, tf.float32) * tf.cast(
                    res2 > eta, tf.float32
                )

                res = cast1 * res1 + cast2 * res2
                res = tf.math.minimum(res, tf.ones(tf.shape(res)))
                res = tf.cast(eta <= eta_s, tf.float32) * res + tf.cast(
                    eta > eta_s, tf.float32
                ) * tf.ones(tf.shape(res))
                res = res.eval(session=tf.compat.v1.Session())

                res = res.squeeze()
                pts = pts.squeeze()

                np.savetxt("psi_pred.txt", res)

                func = interp1d(pts, res)

                self.model.data.train_aux_vars = func(self.model.data.train_x)

                plt.plot(pts, res, label="aux")
                plt.xlim(lim, eta_s)
                plt.legend()
                plt.show()


def psi(x):
    return np.minimum(5 * x, 1)


def pde(x, y, psi):
    eta = x
    q = y[:, 0:1]
    theta_ = y[:, 1:2]
    qp = dde.grad.jacobian(y, x, i=0)
    theta_p = dde.grad.jacobian(y, x, i=1)
    qpp = dde.grad.hessian(y, x, i=0, j=0, component=0)
    theta_pp = dde.grad.hessian(y, x, i=0, j=0, component=1)

    phi = (q - 1) / kappa
    iota = phi + 1 / 2 * kappa * phi**2

    sigma_eta_eta = (psi - eta) * sigma / (1 - (psi - eta) * qp / q)
    sigma_q = qp / q * sigma_eta_eta
    sigma_theta_theta = -theta_p * sigma_eta_eta

    mu_eta_eta_theta = -(psi - eta) * (sigma + sigma_q) * (
        theta_ * (sigma + sigma_q) + sigma_theta_theta
    ) + eta * theta_ * ((a - iota) / q + (1 - psi) * (delta_ - delta))
    mu_q_theta = theta_ * (
        r - (a - iota) / q - phi + delta - sigma * sigma_q
    ) - sigma_theta_theta * (sigma + sigma_q)
    mu_theta = rho - r

    qpp_eq = qpp * sigma_eta_eta**2 * theta_ - 2 * (
        mu_q_theta * q - qp * mu_eta_eta_theta
    )
    theta_pp_eq = sigma_eta_eta**2 * (2 * theta_p**2 - theta_ * theta_pp) - 2 * (
        (rho - r) * theta_**2 + theta_p * mu_eta_eta_theta
    )
    q_increase = tf.minimum(qp, 0)
    theta_hat_increase = tf.minimum(theta_p, 0)

    eta_s = eta_s_var
    ind = tf.cast(x < eta_s, tf.float64)
    return [eq2 * ind, eq3 * ind, q_increase * ind, theta_hat_increase * ind]


geom = dde.geometry.Interval(lim, 1)


def boundary_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], lim)


bc_q = dde.icbc.DirichletBC(geom, lambda x: q_, boundary_l, component=0)

qq = np.loadtxt("q.txt", dtype=float)
thetatheta = np.loadtxt("theta.txt", dtype=float)
eta = qq[:, 0]
q = qq[:, 1]
theta = thetatheta[:, 1]
theta = theta * 1 / theta[-1]

q = np.reshape(q, (-1, 1))
theta_ = 1 / theta
theta_ = np.reshape(theta_, (-1, 1))

q = q.squeeze()
theta_ = theta_.squeeze()

# Make interp1d work from 0 to 1 (the values later become useless)
eta = np.append(eta, 1)
q = np.append(q, 1)
theta_ = np.append(theta_, 1)

q_func = interp1d(eta, q)
theta_func = interp1d(eta, theta_)


def func(x):  # Not the real solution
    return np.hstack((x, x))


def q_l2_error(y_true, y_pred):
    pts = y_true[:, 0]
    ind = pts < eta_s_real
    q_true = q_func(pts) * ind
    q_pred = y_pred[:, 0] * ind

    return np.linalg.norm(q_true - q_pred) / np.linalg.norm(q_true)


def theta_l2_error(y_true, y_pred):
    pts = y_true[:, 1]
    ind = pts < eta_s_real
    theta_true = theta_func(pts) * ind
    theta_pred = y_pred[:, 1] * ind

    return np.linalg.norm(theta_true - theta_pred) / np.linalg.norm(theta_true)


data = dde.data.PDE(
    geom,
    pde,
    [bc_q],
    num_domain=1000,
    num_boundary=2,
    num_test=1000,
    auxiliary_var_function=psi,
    solution=func,
)

# Capture steep increase in q
left_side = dde.geometry.Interval(0, 1e-4)
X = left_side.random_points(10)
data.add_anchors(X)


def input_transform(x):
    return tf.concat(
        (x, 2 * x, 3 * x, 4 * x, 5 * x, 6 * x, 7 * x, 8 * x, 9 * x, 10 * x), axis=1
    )


def output_transform(x, y):
    q = y[:, 0:1]
    theta_ = y[:, 1:2]

    qp = dde.grad.jacobian(y, x, i=0)
    eta_s = eta_s_var

    return tf.concat(
        [
            (x - eta_s) ** 2 * q + C,
            (x - eta_s) ** 2 * x * theta_ - x**2 / eta_s**2 + 2 * x / eta_s,
        ],
        axis=1,
    )


net = dde.nn.FNN([1] + [128] * 6 + [2], "swish", "Glorot uniform")
net.apply_feature_transform(input_transform)
net.apply_output_transform(output_transform)
model = dde.Model(data, net)

lw = [1e6, 1e6, 1e6, 1e6, 1]
model.compile(
    "adam",
    lr=1e-3,
    decay=("inverse time", 1500, 0.5),
    loss_weights=lw,
    external_trainable_variables=[C, eta_s_var],
    metrics=[q_l2_error, theta_l2_error],
)
variable = dde.callbacks.VariableValue([C, eta_s_var], period=1000, precision=9)

update = Update(data.train_x, eta_s_var)
checkpointer = dde.callbacks.ModelCheckpoint(
    "model/model", verbose=1, save_better_only=False, period=2000
)

losshistory, train_state = model.train(
    epochs=100000, callbacks=[update, variable, checkpointer], display_every=1000
)
dde.saveplot(losshistory, train_state, issave=False, isplot=True)

model.compile(
    "adam",
    lr=1e-5,
    decay=("inverse time", 1500, 0.5),
    loss_weights=lw,
    external_trainable_variables=[C, eta_s_var],
    metrics=[q_l2_error, theta_l2_error],
)
losshistory, train_state = model.train(
    epochs=200000,
    callbacks=[update, variable, checkpointer],
    model_restore_path="model/model-100000.ckpt",
    display_every=1000,
)
dde.saveplot(losshistory, train_state, issave=False, isplot=True)

save_data()
