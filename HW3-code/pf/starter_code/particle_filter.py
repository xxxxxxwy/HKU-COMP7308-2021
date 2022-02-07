import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from read_data import read_world, read_sensor_data

# add random seed for generating comparable pseudo random numbers
np.random.seed(123)

# plot preferences, interactive plotting mode
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()


def plot_state(particles, landmarks, map_limits):
    # Visualizes the state of the particle filter.
    #
    # Displays the particle cloud, mean position and landmarks.

    xs = []
    ys = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])

    # landmark positions
    lx = []
    ly = []

    for i in range(len(landmarks)):
        lx.append(landmarks[i + 1][0])
        ly.append(landmarks[i + 1][1])

    # mean pose as current estimate
    estimated_pose = mean_pose(particles)

    # plot filter state
    plt.clf()
    plt.plot(xs, ys, 'r.')
    plt.plot(lx, ly, 'bo', markersize=10)
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy',
               scale_units='xy')
    plt.axis(map_limits)

    plt.pause(0.01)

    return sum(xs) / len(xs), sum(ys) / len(ys)


def initialize_particles(num_particles, map_limits):
    # randomly initialize the particles inside the map limits

    particles = []

    for i in range(num_particles):
        particle = dict()

        # draw x,y and theta coordinate from uniform distribution
        # inside map limits
        particle['x'] = np.random.uniform(map_limits[0], map_limits[1])
        particle['y'] = np.random.uniform(map_limits[2], map_limits[3])
        particle['theta'] = np.random.uniform(-np.pi, np.pi)

        particles.append(particle)

    return particles


def mean_pose(particles):
    # calculate the mean pose of a particle set.
    #
    # for x and y, the mean position is the mean of the particle coordinates
    #
    # for theta, we cannot simply average the angles because of the wraparound
    # (jump from -pi to pi). Therefore, we generate unit vectors from the
    # angles and calculate the angle of their average

    # save x and y coordinates of particles
    xs = []
    ys = []

    # save unit vectors corresponding to particle orientations
    vxs_theta = []
    vys_theta = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])

        # make unit vector from particle orientation
        vxs_theta.append(np.cos(particle['theta']))
        vys_theta.append(np.sin(particle['theta']))

    # calculate average coordinates
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    mean_theta = np.arctan2(np.mean(vys_theta), np.mean(vxs_theta))

    return [mean_x, mean_y, mean_theta]


def sample_motion_model(odometry, particles):
    # Samples new particle positions, based on old positions, the odometry
    # measurements and the motion noise
    # (probabilistic motion models slide)

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    noise = [0.1, 0.1, 0.05, 0.05]
    # noise = [0.2, 0.2, 0.1, 0.1]
    # noise = [0.5, 0.5, 0.3, 0.3]
    # noise = [0.8, 0.8, 0.6, 0.6]
    # generate new particle set after motion update
    new_particles = []

    '''your code here'''
    '''***        ***'''
    sigma_delta_rot1 = noise[0] * abs(delta_rot1) + noise[1] * delta_trans
    sigma_delta_trans = noise[2] * delta_trans + noise[3] * (abs(delta_rot1) + abs(delta_rot2))
    sigma_delta_rot2 = noise[0] * abs(delta_rot2) + noise[1] * delta_trans

    for particle in particles:
        new_particle = dict()
        noisy_delta_rot1 = delta_rot1 + np.random.normal(0, sigma_delta_rot1)
        noisy_delta_trans = delta_trans + np.random.normal(0, sigma_delta_trans)
        noisy_delta_rot2 = delta_rot2 + np.random.normal(0, sigma_delta_rot2)

        new_particle['x'] = particle['x'] + noisy_delta_trans * np.cos(particle['theta'] + noisy_delta_rot1)
        new_particle['y'] = particle['y'] + noisy_delta_trans * np.sin(particle['theta'] + noisy_delta_rot1)
        new_particle['theta'] = particle['theta'] + noisy_delta_rot1 + noisy_delta_rot2
        new_particles.append(new_particle)

    return new_particles


def eval_sensor_model(sensor_data, particles, landmarks):
    # Computes the observation likelihood of all particles, given the
    # particle and landmark positions and sensor measurements
    # (probabilistic sensor models slide)
    #
    # The employed sensor model is range only.

    sigma_r = 0.2

    # measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']

    weights = []

    '''your code here'''
    '''***        ***'''
    for p in particles:
        wp = 1.0
        for i, id in enumerate(ids):
            mx, my = landmarks[id][0], landmarks[id][1]
            px, py = p['x'], p['y']
            e_range = np.sqrt(np.square(mx - px) + np.square(my - py))
            m_range = ranges[i]
            likelihood = scipy.stats.norm.pdf(m_range, e_range, sigma_r)
            wp = wp * likelihood
        weights.append(wp)

    # normalize weights
    normalizer = sum(weights)
    weights = weights / normalizer
    return weights


# bearing
def eval_sensor_model_bearing(sensor_data, particles, landmarks):
    # Computes the observation likelihood of all particles, given the
    # particle and landmark positions and sensor measurements
    # (probabilistic sensor models slide)
    #
    # The employed sensor model is range only.

    sigma_r = 0.2
    # sigma_r = 0.5
    # sigma_r = 0.9

    # measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']
    bearings = sensor_data['bearing']
    weights = []

    '''your code here'''
    '''***        ***'''
    for p in particles:
        wp = 1.0
        for i, id in enumerate(ids):
            mx, my = landmarks[id][0], landmarks[id][1]
            px, py = p['x'], p['y']

            range_e = np.sqrt(np.square(mx - px) + np.square(my - py))
            bearing_e = np.arctan2(my - py, mx - px) - p['theta']

            measurement_range = ranges[i]
            measurement_bearing = bearings[i]

            angle1 = measurement_bearing
            angle2 = bearing_e
            dif_bearing = np.arctan2(np.sin(angle1 - angle2), np.cos(angle1 - angle2))

            likelihood_r = scipy.stats.norm.pdf(np.abs(measurement_range - range_e), 0, sigma_r)
            likelihood_b = scipy.stats.norm.pdf(dif_bearing, 0, sigma_r)

            wp = wp * likelihood_r * likelihood_b

        weights.append(wp)

    # normalize weights
    normalizer = sum(weights)
    weights = weights / normalizer
    return weights


def resample_particles(particles, weights):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle weights.

    new_particles = []

    '''your code here'''
    '''***        ***'''
    c = weights[0]
    step = 1.0 / len(particles)
    u = np.random.uniform(0, 1.0 / len(particles))
    i = 0
    for p in particles:
        while (u > c):
            i = i + 1
            c += weights[i]
        new_particles.append(particles[i])
        u += step
    return new_particles


def KLD(particles, weights):
    new_particles = []
    z_1_minus_delta = 0.5040

    bins = set()
    for i in particles:
        bins_size_xy = 50
        bins_size_theta = 50
        x_hash = int(i['x'] * bins_size_xy)
        y_hash = int(i['y'] * bins_size_xy)
        theta_hash = int(i['theta'] * bins_size_theta)
        t_hash = int(x_hash * 1e8 + y_hash * 1e4 + theta_hash)
        bins.add(t_hash)

    k = len(bins)
    k = 2 if k < 2 else k

    n = (k - 1) / (2 * 0.2) * (1 - 2 / (9 * (k - 1)) + z_1_minus_delta * (2 / (9 * (k - 1))) ** 0.5) ** 3

    sum_w = np.array([sum(weights[0:i + 1]) for i in range(len(weights))])

    for _ in range(int(n)):
        i = np.argwhere(sum_w > np.random.uniform(0, 1))[0][0]
        new_particles.append(particles[i])

    return new_particles


def main_pro():
    # implementation of a particle filter for robot pose estimation

    print("Reading landmark positions")
    landmarks = read_world("../data/world.dat")

    print("Reading sensor data")
    sensor_readings = read_sensor_data("../data/sensor_data.dat")

    # initialize the particles
    map_limits = [-1, 12, 0, 10]
    particles = initialize_particles(1000, map_limits)

    xx = []
    yy = []
    # run particle filter
    for timestep in range(int(len(sensor_readings) / 2)):
        # plot the current state
        x, y = plot_state(particles, landmarks, map_limits)
        # print(x,y)
        # predict particles by sampling from motion model with odometry info
        new_particles = sample_motion_model(sensor_readings[timestep, 'odometry'], particles)

        # calculate importance weights according to sensor model
        weights = eval_sensor_model_bearing(sensor_readings[timestep, 'sensor'], new_particles, landmarks)
        # weights = eval_sensor_model(sensor_readings[timestep, 'sensor'], new_particles, landmarks)

        # resample new particle set according to their importance weights
        # particles = resample_particles(new_particles, weights)
        particles = KLD(new_particles, weights)
        xx.append(x)
        yy.append(y)

    return xx, yy


if __name__ == "__main__":
    x, y = main_pro()
    plt.scatter(x, y)
    plt.show()
    plt.pause(1000)
