import numpy as np

np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
import copy


class GridWorld(object):
    """
    Gridworld object defines how the robot senses and moves within
    the grid map
    """

    def __init__(self):
        """
        Initializes robot occupancy map, current position/state, action failure
        probability and sensor failure possibility
        
        Arguments:
        self.occupancyMap -- map of the occupancy grid in the image above. 3x6
        numpy array showing occupied cells
        self.map -- map showing the current position of the robot in the map
        self.state -- current position of the robot within the grid map. [x,y]
        self.actionFailsProb -- probability the action fails 
        self.state -- probability the sensor fails
        """
        self.map = np.zeros((3, 6), dtype=bool)

        self.occupancyMap = np.array([[1, 1, 0, 0, 0, 1],
                                      [0, 0, 1, 1, 0, 1],
                                      [0, 0, 1, 0, 1, 0]], dtype=bool)

        self.state = np.array([1, 0])
        self.actionFailsProb = 1e-5
        self.senseFailsProb = 0.2

    def step(self, action):
        """
        Defines the rules for moving within the grid
        
        Arguments:
        self -- (1, 2) numpy array of the robot's current position in the grid
        action -- the direction for the robot to move e.g. "r" for right
        
        Returns:
        updates self.state after the action
        """
        actionFails = np.random.rand() < self.actionFailsProb

        if actionFails:
            pass
        else:
            '''your code here'''
            if action == "r":
                self.state[1] = np.minimum(self.state[1] + 1, self.map.shape[1] - 1)
            elif action == "l":
                self.state[1] = np.maximum(self.state[1] - 1, 0)
            elif action == "u":
                self.state[0] = np.maximum(self.state[0] - 1, 0)
            elif action == "d":
                self.state[0] = np.minimum(self.state[0] + 1, self.map.shape[0] - 1)
            '''***        ***'''

    def sense(self):
        """
        Defines the rules for sensing with a probability the sensor fails
        """

        # Implement the probability the sensor fails

        '''your code here'''
        senseFails = np.random.binomial(n=1, p=self.senseFailsProb, size=1)
        '''***        ***'''

        trueSense = self.occupancyMap[self.state[0], self.state[1]]
        if senseFails:
            return not (trueSense)
        else:
            return trueSense


class StateEstimator(object):
    """
    Contains the methods required to update the belief based on the 
    probability the action or sensor fails
    
    """

    def __init__(self, world):
        """
        Initializes the object, selecting either the occupancy map or grid map          
        """

        self.actionFailsProb = world.actionFailsProb
        self.senseFailsProb = world.senseFailsProb

        # Using the occupancy map         
        self.map = world.occupancyMap

        # Initialize belief matrix, normalized and equal probability for each cell

        # self.belief = np.ones(world.map.shape) / np.prod(world.map.shape)
        self.belief = np.array([[0, 0, 0.1, 0.1, 0.1, 0],
                                [0.1, 0.1, 0, 0, 0.1, 0],
                                [0.1, 0.1, 0, 0.1, 0, 0.1]], dtype=float)

    def actionUpdate(self, action):
        """
        Calculates the new belief for each action based on the probability the action might fail
        
        Variables:
        self.belief -- belief from the previous time step
        newBelief -- updated belief
        action -- string which states the direction the robot moves
        """

        newBelief = np.zeros(self.belief.shape, dtype=float)

        '''your code here'''
        if action == "r":
            # col 0
            newBelief[:, 0] = self.belief[:, 0] * self.actionFailsProb
            # col 1-4
            newBelief[:, 1:-1] = (
                    self.belief[:, 1:-1] * self.actionFailsProb + self.belief[:, 0:-2] * (1 - self.actionFailsProb))
            # col 5
            newBelief[:, -1] = (self.belief[:, -1] * 1 + self.belief[:, -2] * (1 - self.actionFailsProb))

        if action == "l":
            newBelief[:, -1] = self.belief[:, -1] * self.actionFailsProb
            newBelief[:, 1:-1] = (
                    self.belief[:, 1:-1] * self.actionFailsProb + self.belief[:, 2:] * (1 - self.actionFailsProb))
            newBelief[:, 0] = (self.belief[:, 0] * 1 + self.belief[:, 1] * (1 - self.actionFailsProb))

        if action == "u":
            newBelief[-1, :] = self.belief[-1, :] * self.actionFailsProb
            newBelief[1, :] = (
                    self.belief[1, :] * self.actionFailsProb + self.belief[-1, :] * (1 - self.actionFailsProb))
            newBelief[0, :] = self.belief[0, :] * 1 + self.belief[1, :] * (1 - self.actionFailsProb)

        if action == "d":
            newBelief[0, :] = self.belief[0, :] * self.actionFailsProb
            newBelief[1, :] = (
                    self.belief[1, :] * self.actionFailsProb + self.belief[0, :] * (1 - self.actionFailsProb))
            newBelief[-1, :] = self.belief[-1, :] * 1 + self.belief[1, :] * (1 - self.actionFailsProb)
        '''***        ***'''

        self.belief = newBelief

    def measurementUpdate(self, measurement):
        """
        Updates the measurement based on Bayes' filter
        
        Variables:
        prior -- previous belief
        posterior -- new belief
        measurement -- the most recent measurement
        posterior = prior*likelihood then normalize
        belief = posterior
        """

        prior = self.belief

        '''your code here'''
        likelihood = (self.senseFailsProb * np.not_equal(measurement, self.map, dtype=int) + (
                1 - self.senseFailsProb) * np.equal(measurement, self.map, dtype=int))
        posterior = prior * likelihood
        marginal = np.sum(posterior)
        self.belief = posterior / marginal
        '''***        ***'''


def main():
    world = GridWorld()
    positionMap = copy.copy(world.map)
    positionMap[world.state[0], world.state[1]] = 1
    estimator = StateEstimator(world)

    print(estimator.belief)

    # Using the occupancy map, move the robot right 3, down 1, and left 1
    for indexAction, action in enumerate(["r", "r", "r", "d", "l"]):
        measurement = world.sense()
        estimator.measurementUpdate(measurement)

        world.step(action)
        estimator.actionUpdate(action)

        print("Estimate after move %d:" % (indexAction + 1))
        print("\n")
        ax = plt.gca()
        ax.set_yticks(np.arange(0, 3, 1));
        extent = (0, positionMap.shape[1], positionMap.shape[0], 0)
        plt.imshow(estimator.belief, extent=extent, cmap='binary', interpolation='nearest')
        plt.grid(color='w', linestyle='-', linewidth=2)
        plt.show()
        print(estimator.belief)
        print("\n")


if __name__ == "__main__":
    main()
