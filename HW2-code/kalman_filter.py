import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from read_data import read_world, read_sensor_data
from matplotlib.patches import Ellipse

# plot preferences, interactive plotting mode
fig = plt.figure()
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()


def plot_state(mu, sigma, landmarks, map_limits):
    # Visualizes the state of the kalman filter.
    #
    # Displays the mean and standard deviation of the belief,
    # the state covariance sigma and the position of the 
    # landmarks.

    # landmark positions
    lx = []
    ly = []

    for i in range(len(landmarks)):
        lx.append(landmarks[i + 1][0])
        ly.append(landmarks[i + 1][1])

    # mean of belief as current estimate
    estimated_pose = mu

    # calculate and plot covariance ellipse
    covariance = sigma[0:2, 0:2]
    eigenvals, eigenvecs = np.linalg.eig(covariance)

    # get largest eigenvalue and eigenvector
    max_ind = np.argmax(eigenvals)
    max_eigvec = eigenvecs[:, max_ind]
    max_eigval = eigenvals[max_ind]

    # get smallest eigenvalue and eigenvector
    min_ind = 0
    if max_ind == 0:
        min_ind = 1

    min_eigvec = eigenvecs[:, min_ind]
    min_eigval = eigenvals[min_ind]

    # chi-square value for sigma confidence interval
    chisquare_scale = 2.2789

    # calculate width and height of confidence ellipse
    width = 2 * np.sqrt(chisquare_scale * max_eigval)
    height = 2 * np.sqrt(chisquare_scale * min_eigval)
    angle = np.arctan2(max_eigvec[1], max_eigvec[0])

    # generate covariance ellipse
    ell = Ellipse(xy=[estimated_pose[0], estimated_pose[1]], width=width, height=height, angle=angle / np.pi * 180)
    ell.set_alpha(0.25)

    # plot filter state and covariance
    plt.clf()
    plt.gca().add_artist(ell)
    plt.plot(lx, ly, 'bo', markersize=10)
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy',
               scale_units='xy')
    plt.axis(map_limits)

    plt.pause(0.01)


def prediction_step(odometry, mu, sigma):
    # Updates the belief, i.e., mu and sigma, according to the motion 
    # model
    # 
    # mu: 3x1 vector representing the mean (x,y,theta) of the 
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution 

    x = mu[0]
    y = mu[1]
    theta = mu[2]

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    '''your code here'''
    '''***        ***'''
    G_t = np.array([[1, 0, -delta_trans * np.sin(theta + delta_rot1)],
                    [0, 1, delta_trans * np.cos(theta + delta_rot1)],
                    [0, 0, 1]])
    k = 1.0
    R_t = np.array([[0.2, 0, 0],
                    [0, 0.2, 0],
                    [0, 0, 0.02]]) * k
    mu_t = np.array([x + delta_trans * np.cos(theta + delta_rot1),
                     y + delta_trans * np.sin(theta + delta_rot1),
                     theta + delta_rot1 + delta_rot2])
    sigma_t = G_t.dot(sigma).dot(G_t.T) + R_t

    return mu_t, sigma_t


def correction_step(sensor_data, mu, sigma, landmarks):
    # updates the belief, i.e., mu and sigma, according to the
    # sensor model
    # 
    # The employed sensor model is range-only
    #
    # mu: 3x1 vector representing the mean (x,y,theta) of the 
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution 

    x = mu[0]
    y = mu[1]
    theta = mu[2]

    # measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']

    '''your code here'''
    '''***        ***'''
    h = np.zeros(len(ids))
    H_t = np.zeros((len(ids), 3))
    i = 0
    for id in ids:
        h[i] = np.sqrt((x - landmarks[id][0]) ** 2 + (y - landmarks[id][1]) ** 2)
        H_t[i] = [(x - landmarks[id][0]) / h[i], (y - landmarks[id][1]) / h[i], 0]
        i += 1
    k = 1.0
    Q_t = np.eye(len(ids)) * 0.5 * k
    K_t = sigma.dot(H_t.T).dot(np.linalg.inv(H_t.dot(sigma).dot(H_t.T) + Q_t))
    mu_t = mu + K_t.dot(ranges - h)
    sigma_t = (np.eye(3) - K_t.dot(H_t)).dot(sigma)

    return mu_t, sigma_t


def correction_step_bearing(sensor_data, mu, sigma, landmarks):
    # updates the belief, i.e., mu and sigma, according to the
    # sensor model
    #
    # The employed sensor model is range-only
    #
    # mu: 3x1 vector representing the mean (x,y,theta) of the
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution

    x = mu[0]
    y = mu[1]
    theta = mu[2]

    # measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']
    bearing = sensor_data['bearing']

    '''your code here'''
    '''***        ***'''
    Q_t = np.eye(len(ids) * 2) * 0.5
    H_t, Z_t, h = [], [], []

    for i in range(len(ids)):
        lx = landmarks[ids[i]][0]
        ly = landmarks[ids[i]][1]
        q = (lx - x) ** 2 + (ly - y) ** 2
        hi = [np.arctan2([ly - y], [lx - x])[0] - theta, q ** 0.5]
        Hi = [[(ly - y) / q, (x - lx) / q, -1], [-(lx - x) / np.sqrt(q), -(ly - y) / np.sqrt(q), 0]]
        H_t.append(Hi[0])
        H_t.append(Hi[1])
        Z_t.append(bearing[i])
        Z_t.append(ranges[i])
        h.append(hi[0])
        h.append(hi[1])
    H_t, Z_t, h = np.array(H_t), np.array(Z_t), np.array(h)
    inv = np.linalg.inv(np.dot(np.dot(H_t, sigma), np.transpose(H_t)) + Q_t)
    K_t = np.dot(np.dot(sigma, np.transpose(H_t)), inv)
    mu_t = mu + np.dot(K_t, (Z_t - h))
    sigma_t = np.dot(np.eye(len(K_t)) - np.dot(K_t, H_t), sigma)

    return mu_t, sigma_t


def main():
    try:
        # implementation of an extended Kalman filter for robot pose estimation

        print("Reading landmark positions")
        landmarks = read_world("../data/world.dat")

        print("Reading sensor data")
        sensor_readings = read_sensor_data("../data/sensor_data.dat")

        # initialize belief
        mu = [0.0, 0.0, 0.0]
        sigma = np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]])

        map_limits = [-1, 12, -1, 10]

        # run kalman filter
        for timestep in range(int(len(sensor_readings) / 2)):
            # plot the current state
            plot_state(mu, sigma, landmarks, map_limits)

            # perform prediction step
            mu, sigma = prediction_step(sensor_readings[timestep, 'odometry'], mu, sigma)

            # perform correction step
            mu, sigma = correction_step(sensor_readings[timestep, 'sensor'], mu, sigma, landmarks)

        plt.show('hold')
    except:
        return


if __name__ == "__main__":
    main()
