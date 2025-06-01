import numpy as np
import argparse

import utils
import read_imu
import read_wheels
import read_gps
import read_ground_truth

TRUNCATION_END = -1
KALMAN_FILTER_RATE = 1

R_GPS = np.eye(2) * np.power(10, 2)

Q = np.diag([1.0, 1.0])  # 浮点数，大小 (2,2)

ROBOT_WIDTH_WHEEL_BASE = 0.562356

LABEL_ESTIMATION_TYPE = f"Estimated - UKF Wheels with GPS {KALMAN_FILTER_RATE}Hz (XY Only)"

def wraptopi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def find_nearest_index(array: np.ndarray, time):
    diff_arr = array - time
    idx = np.where(diff_arr <= 0, diff_arr, -np.inf).argmax()
    return idx

def generate_sigma_points(x, P, alpha=0.1, beta=2, kappa=0):
    n = len(x)
    lambda_ = alpha**2 * (n + kappa) - n
    sigma_points = np.zeros((2 * n + 1, n))
    Wm = np.full(2 * n + 1, 1 / (2 * (n + lambda_)))
    Wc = np.copy(Wm)
    Wm[0] = lambda_ / (n + lambda_)
    Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
    sqrt_P = np.linalg.cholesky((n + lambda_) * P)
    sigma_points[0] = x
    for i in range(n):
        sigma_points[i + 1] = x + sqrt_P[:, i]
        sigma_points[n + i + 1] = x - sqrt_P[:, i]
    return sigma_points, Wm, Wc

def propagate_motion_model(sigma_points, dt, vl, vr, theta):
    new_sigma = []
    v_c = 0.5 * (vl + vr)
    theta_new = wraptopi(theta)
    for sp in sigma_points:
        x, y = sp
        x_new = x + v_c * np.cos(theta_new) * dt
        y_new = y + v_c * np.sin(theta_new) * dt
        new_sigma.append([x_new, y_new])
    return np.array(new_sigma)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filedate')
    args = parser.parse_args()
    FILE_DATE = args.filedate

    gps_data = read_gps.read_gps(FILE_DATE, False)
    imu_data = read_imu.read_imu(FILE_DATE)
    euler_data = read_imu.read_euler(FILE_DATE)
    wheel_data = read_wheels.read_wheels(FILE_DATE)
    ground_truth = read_ground_truth.read_ground_truth(FILE_DATE, truncation=TRUNCATION_END)
    ground_truth = ground_truth[:TRUNCATION_END, :]
    gps_data = gps_data[:TRUNCATION_END, :]
    imu_data = imu_data[:TRUNCATION_END, :]
    euler_data = euler_data[:TRUNCATION_END, :]
    wheel_data = wheel_data[:TRUNCATION_END, :]

    x_true = ground_truth[:, 1]
    y_true = ground_truth[:, 2]
    theta_true = ground_truth[:, 3]
    true_times = ground_truth[:, 0]

    dt = 1 / KALMAN_FILTER_RATE
    t = np.arange(ground_truth[0, 0], ground_truth[-1, 0], dt)
    N = len(t)

    x_est = np.zeros([N, 2])  # 只有 (x, y)
    P_est = np.zeros([N, 2, 2])

    x_est[0] = np.array([x_true[0], y_true[0]])
    P_est[0] = np.diag([0.5, 0.5])

    x_true_arr = np.zeros(N)
    y_true_arr = np.zeros(N)
    theta_true_arr = np.zeros(N)

    x_true_arr[0] = x_true[0]
    y_true_arr[0] = y_true[0]
    theta_true_arr[0] = theta_true[0]

    omega = imu_data[:, 3]
    theta_imu = euler_data[:, 1]

    gps_x = gps_data[:, 1]
    gps_y = gps_data[:, 2]

    v_left_wheel = wheel_data[:, 2]
    v_right_wheel = wheel_data[:, 3]

    gps_times = gps_data[:, 0]
    wheel_times = wheel_data[:, 0]
    imu_times = imu_data[:, 0]
    euler_times = euler_data[:, 0]

    gps_counter = 0
    wheel_counter = 0
    imu_counter = 0
    euler_counter = 0
    ground_truth_counter = 0
    prev_gps_counter = -1

    for k in range(1, N):
        print(k, "/", N)
        imu_counter = find_nearest_index(imu_times, t[k])
        euler_counter = find_nearest_index(euler_times, t[k])
        gps_counter = find_nearest_index(gps_times, t[k])
        wheel_counter = find_nearest_index(wheel_times, t[k])
        ground_truth_counter = find_nearest_index(true_times, t[k])

        sigma_points, Wm, Wc = generate_sigma_points(x_est[k-1], P_est[k-1])
        sigma_points_pred = propagate_motion_model(
            sigma_points,
            dt,
            v_left_wheel[wheel_counter],
            v_right_wheel[wheel_counter],
            theta_imu[euler_counter]
        )

        x_predicted = np.dot(Wm, sigma_points_pred)
        P_predicted = Q.copy()
        for i in range(len(sigma_points)):
            dx = sigma_points_pred[i] - x_predicted
            P_predicted += Wc[i] * np.outer(dx, dx)

        if gps_counter != prev_gps_counter:
            z_sigma = sigma_points_pred[:, 0:2]  # 观测是 (x, y)
            z_pred = np.dot(Wm, z_sigma)

            P_zz = R_GPS.copy()
            for i in range(len(sigma_points)):
                dz = z_sigma[i] - z_pred
                P_zz += Wc[i] * np.outer(dz, dz)

            P_xz = np.zeros((2, 2))
            for i in range(len(sigma_points)):
                dx = sigma_points_pred[i] - x_predicted
                dz = z_sigma[i] - z_pred
                P_xz += Wc[i] * np.outer(dx, dz)

            K = np.dot(P_xz, np.linalg.inv(P_zz))
            z_meas = np.array([gps_x[gps_counter], gps_y[gps_counter]])
            x_predicted = x_predicted + np.dot(K, (z_meas - z_pred))
            P_predicted = P_predicted - K @ P_zz @ K.T

            prev_gps_counter = gps_counter

        x_est[k] = x_predicted
        P_est[k] = P_predicted

        x_true_arr[k] = x_true[ground_truth_counter]
        y_true_arr[k] = y_true[ground_truth_counter]
        theta_true_arr[k] = theta_true[ground_truth_counter]

    print('Done! Plotting now.')
    utils.export_to_kml(x_est[:,0], x_est[:,1], x_true_arr, y_true_arr, LABEL_ESTIMATION_TYPE, "Ground Truth", FILE_DATE)
    utils.save_results(x_est, P_est, x_true_arr, y_true_arr, theta_true_arr, t, f"{FILE_DATE}_{LABEL_ESTIMATION_TYPE}")
    utils.plot_position_comparison_2D(x_est[:,0], x_est[:,1], x_true_arr, y_true_arr, LABEL_ESTIMATION_TYPE, "Ground Truth", FILE_DATE)
