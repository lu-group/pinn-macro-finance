import os

os.environ["DDEBACKEND"] = "tensorflow"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"

import numpy as np

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import bisect
from deepxde import utils
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

import deepxde as dde
from deepxde import utils
from deepxde.backend import tf
from deepxde.callbacks import Callback
from deepxde.callbacks import OperatorPredictor

dde.config.set_default_float("float64")
dde.config.disable_xla_jit()

a_ = 0.05
rho = 0.06
r = 0.05
sigma = 0.025
delta = 0.03
delta_ = 0.08
kappa = 10
eta_s_train = dde.Variable(
    0.0, dtype=tf.float64
)  # eta* = 0.2 * sigmoid(eta* var) + 0.3
eta_s_ref = 0.364762616462568  # Reference value -- not used in training
q_ = 0.48616429  # Calculated prior to training
C = dde.Variable(1.0, dtype=tf.float64)  # Represents q(eta*)

a_true = 1.1
a_init = np.random.random_sample() + a_true - 0.5  # From 0.6 to 1.6
a_train = dde.Variable(a_init, dtype=tf.float64)  # a = a_train / 10


class Update(OperatorPredictor):
    def __init__(self, x, eta_s_train, a):
        # Calculates these values that are used in the callback
        def info(x, y):
            return (
                dde.grad.jacobian(y, x, i=0, j=0),  # dq/d(eta)
                dde.grad.jacobian(y, x, i=1, j=0),  # d(theta)/d(eta)
                eta_s_train,
                a,
            )

        super().__init__(x, info)

    def on_train_begin(self):
        self.on_predict_end()
        self.file.flush()

    def on_batch_begin(self):
        pts = np.linspace(0, 1, 10000)
        pts = np.reshape(pts, (-1, 1))

        y = self.model._outputs(False, pts)
        q = y[:, 0]
        theta_ = y[:, 1]

        qp, theta_p, eta_s_train, a = utils.to_numpy(self.tf_op(pts))

        eta = pts
        eta_s = 1 / ((1 + np.exp(-eta_s_train)) * 5) + 0.3
        ind = eta < eta_s

        epoch_num = self.model.train_state.epoch

        # Every 100 epochs, update psi if q is increasing
        if epoch_num % 1000 == 0 and epoch_num >= 1000:
            qp_sum = np.sum(np.minimum(0, qp * ind))
            if np.isclose(qp_sum, 0):

                def F(eta, q, theta_, qp, theta_p):
                    if eta > eta_s:
                        return 1.0
                    if np.isclose(eta, 0) or np.isclose(theta_, 0) or np.isclose(q, 0):
                        return 0.0
                    if qp < 0:
                        return 1.0

                    a = a_train / 10

                    def eq(psi):
                        sigma_eta_eta = (
                            (psi - eta) * sigma / (1 - (psi - eta) * qp / q)
                        )  # Otherwise the denominator is exactly 0 somehow
                        sigma_q = qp / q * sigma_eta_eta
                        sigma_theta = -theta_p / theta_ * sigma_eta_eta
                        return (
                            (a - a_) / q
                            + delta_
                            - delta
                            + (sigma + sigma_q) * sigma_theta
                        )

                    # Use bisection algorithm to solve for psi
                    lb = eta + 1e-6
                    if qp == 0:  # Occurs at eta = eta*, just make rb = inf
                        rb = 1e9
                    else:
                        rb = eta + q / qp - 1e-6
                    try:
                        res = bisect(eq, lb, rb)
                    except:
                        if eta < 0.1:
                            return 0.0
                        else:
                            return 1.0
                    if res > 1:
                        return 1.0
                    return res

                eta = np.reshape(eta, (-1, 1))
                q = np.reshape(q, (-1, 1))
                theta_ = np.reshape(theta_, (-1, 1))
                qp = np.reshape(qp, (-1, 1))
                theta_p = np.reshape(theta_p, (-1, 1))

                F_vec = np.vectorize(F)
                res = F_vec(pts, q, theta_, qp, theta_p)

                res = res.squeeze()
                pts = pts.squeeze()

                f = IUS(pts, res)
                fp = f.derivative()
                psi_new = f(self.model.data.train_x)
                psip_new = fp(self.model.data.train_x)
                self.model.data.train_aux_vars = np.hstack((psi_new, psip_new))

                # Plot psi every 1000 epochs when it updates
                if epoch_num % 1000 == 0:
                    plt.plot(pts, res, label="aux")
                    plt.xlim(0, eta_s)
                    plt.legend()
                    plt.show()


def psi_info(x):
    # Returns (psi, psi'), initial guess is psi = 1
    return np.hstack((np.ones(x.shape), np.zeros(x.shape)))


def integrate(x, eta_s):
    # Computes integral from 0 to eta*
    amt_nonzero = tf.math.maximum(tf.math.count_nonzero(x, dtype=tf.float64), 1.0)
    return eta_s * tf.reduce_sum(x) / amt_nonzero


def pde(x, y, psi_info):
    eta = x

    q = y[:, 0]
    theta_ = y[:, 1]  # theta_ is theta hat
    qp = dde.grad.jacobian(y, x, i=0)
    theta_p = dde.grad.jacobian(y, x, i=1)
    qpp = dde.grad.hessian(y, x, i=0, j=0, component=0)
    theta_pp = dde.grad.hessian(y, x, i=0, j=0, component=1)

    psi = psi_info[:, 0]
    psip = psi_info[:, 1]

    eta = tf.reshape(eta, [-1])
    qp = tf.reshape(qp, [-1])
    theta_p = tf.reshape(theta_p, [-1])
    qpp = tf.reshape(qpp, [-1])
    theta_pp = tf.reshape(theta_pp, [-1])

    phi = (q - 1) / kappa
    iota = phi + 1 / 2 * kappa * phi**2

    a = a_train / 10
    eta_s = 0.2 * tf.sigmoid(eta_s_train) + 0.3
    ind = tf.cast(eta < eta_s, tf.float64)

    sigma_eta_eta = (psi - eta) * sigma / (1 - (psi - eta) * qp / q)
    sigma_eta_eta_2p = (
        2
        * sigma**2
        * (psi - eta)
        * q
        * (
            q**2 * (psip - 1)
            + (eta - psi) ** 2 * q * qpp
            + (eta - psi) ** 2 * (-(qp**2))
        )
    ) / ((eta - psi) * qp + q) ** 3
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

    # Computes f(eta), which is used in the moment condition
    Q = (2 * mu_eta_eta_theta - sigma_eta_eta_2p * theta_) / tf.math.maximum(
        (sigma_eta_eta**2 * theta_), 1e-9
    )
    Q = tf.math.maximum(Q, 0)

    # Q_avg(eta) = average of Q from eta to eta*
    eta_reshaped = tf.reshape(eta, [-1, 1])
    ind1 = eta_reshaped > eta
    ind2 = eta_reshaped < (tf.ones(tf.shape(eta_reshaped), dtype=tf.float64) * eta_s)
    Q_reshaped = tf.reshape(Q, [-1, 1])
    Q_int_new = tf.cast(ind1, tf.float64) * tf.cast(ind2, tf.float64) * Q_reshaped
    Q_avg = tf.reduce_sum(Q_int_new, axis=0) / tf.math.maximum(
        tf.math.count_nonzero(Q_int_new, axis=0, dtype=tf.float64), 1.0
    )

    A_int = tf.math.exp(-Q_avg * (eta_s - eta))
    A = 1 / integrate(A_int * ind, eta_s)

    f = A * A_int

    # Loss from the a_target moment condition
    a_target_int = (psi * a + (1 - psi) * a_) * f * ind
    a_target_pred = integrate(a_target_int, eta_s)
    a_target = 0.109506136589605
    a_target_loss = (a_target_pred - a_target) * tf.ones(
        tf.shape(eta), dtype=tf.float64
    )

    return [
        qpp_eq * ind,
        theta_pp_eq * ind,
        q_increase * ind,
        theta_hat_increase * ind,
        a_target_loss,
    ]


geom = dde.geometry.Interval(0, 1)


def boundary_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


bc_q = dde.icbc.DirichletBC(geom, lambda x: q_, boundary_l, component=0)

qq = np.loadtxt("q.txt", dtype=float)
thetatheta = np.loadtxt("theta.txt", dtype=float)
eta = qq[:, 0]
q = qq[:, 1]
theta_unnorm = thetatheta[:, 1]
theta = theta_unnorm * 1 / theta_unnorm[-1]  # Satisfy theta(eta*) = 1 condition
theta_ = 1 / theta

q = q.squeeze()
theta_ = theta_.squeeze()

# Make interp1d work from 0 to 1
eta = np.append(eta, 1)
q = np.append(q, 1)
theta_ = np.append(theta_, 1)

q_func = interp1d(eta, q)
theta_func = interp1d(eta, theta_)


def func(x):
    # Not the true function -- tells the L2 error functions what the eta points are
    return np.hstack((x, x))


def q_l2_error(y_true, y_pred):
    eta = y_true[:, 0]

    ind = eta < eta_s_ref
    q_true = q_func(eta) * ind
    q_pred = y_pred[:, 0] * ind

    return np.linalg.norm(q_true - q_pred) / np.linalg.norm(q_true)


def theta_l2_error(y_true, y_pred):
    eta = y_true[:, 1]

    ind = eta < eta_s_ref
    theta_true = theta_func(eta) * ind
    theta_pred = y_pred[:, 1] * ind

    return np.linalg.norm(theta_true - theta_pred) / np.linalg.norm(theta_true)


data = dde.data.PDE(
    geom,
    pde,
    [bc_q],
    num_domain=8190,
    num_boundary=2,
    num_test=8192,
    auxiliary_var_function=psi_info,
    solution=func,
)


def input_transform(x):
    return tf.concat(
        (x, 2 * x, 3 * x, 4 * x, 5 * x, 6 * x, 7 * x, 8 * x, 9 * x, 10 * x), axis=1
    )


# Satisfies all boundary conditions exactly except q(0) = q_
def output_transform(x, y):
    q = y[:, 0:1]
    theta_ = y[:, 1:2]

    eta_s = 0.2 * tf.sigmoid(eta_s_train) + 0.3

    return tf.concat(
        [
            (x - eta_s) ** 2 * q + C,
            (x - eta_s) ** 2 * x * theta_ - x**2 / eta_s**2 + 2 * x / eta_s,
        ],
        axis=1,
    )


net = dde.nn.FNN([1] + [64] * 5 + [2], "swish", "Glorot uniform")
net.apply_feature_transform(input_transform)
net.apply_output_transform(output_transform)
model = dde.Model(data, net)

variable = dde.callbacks.VariableValue(
    [C, eta_s_train, a_train], period=1000, precision=9
)
update = Update(data.train_x, eta_s_train, a_train)

lw = [5e4, 1e4, 1e5, 1e3, 1e4, 1e0]

model.compile(
    "adam",
    lr=1e-4,
    decay=("inverse time", 2000.0, 1.0),
    loss_weights=lw,
    external_trainable_variables=[C, eta_s_train, a_train],
    metrics=[q_l2_error, theta_l2_error],
)

losshistory, train_state = model.train(
    epochs=200000, callbacks=[variable, update], display_every=1000
)

dde.saveplot(losshistory, train_state, issave=False, isplot=True)

q_txt = np.loadtxt("q.txt", dtype=float)
theta_txt = np.loadtxt("theta.txt", dtype=float)

eta_s_train = variable.get_value()[1]
eta_s = 0.2 / (1 + np.exp(-eta_s_train)) + 0.3

eta_true = q_txt[:, 0]
q_true = q_txt[:, 1]
theta_true_unnorm = theta_txt[:, 1]
theta_true = (
    theta_true_unnorm * 1 / theta_true_unnorm[-1]
)  # Satisfy theta(eta*) = 1 condition

eta_pred = np.linspace(0, eta_s, 10000).reshape(10000, 1)
sol_pred = model.predict(eta_pred)
q_pred = sol_pred[:, 0:1]
theta_pred = 1 / sol_pred[:, 1:2]


plt.plot(eta_pred, q_pred, label="q pred")
plt.plot(eta_true, q_true, label="q true")
plt.legend()
plt.xlim(0, 0.4)
plt.ylim(0, 2)
plt.show()

plt.plot(eta_pred, theta_pred, label="theta pred.")
plt.plot(eta_true, theta_true, label="theta true")
plt.xlim(0, 0.4)
plt.ylim(0, 100)
plt.legend()
plt.show()

a_final = variable.get_value()[2] / 10
print("Predicted a: %.5f" % a_final)
print("True a: 0.11")
