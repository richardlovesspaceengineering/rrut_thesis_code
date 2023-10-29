# %%
from Sampling import Sampling
import numpy as np
import matplotlib.pyplot as plt
import warnings


class RandomWalk(Sampling):
    def __init__(self, bounds, num_steps, step_size_pct=0.2):
        """
        Step size must be given as a percentage of (xmax - xmin) for each dimension.
        """
        self.bounds = bounds
        self.num_steps = num_steps
        self.step_size_pct = step_size_pct
        self.dim = self.bounds.shape[1]  # dimensionality of the problem
        self.initialise_step_sizes()

    def initialise_step_sizes(self):
        self.step_size_array = np.zeros(self.dim)
        for i in range(self.dim):
            self.step_size_array[i] = (self.bounds[1,i] - self.bounds[0,i])*self.step_size_pct 
    
    def random_pm(self):
        # correct implementation for a true RW
        return 1 if np.random.random() < 0.5 else -1

    def within_bounds(self, point, dim):
        return self.above_lower_bound(point, dim) and self.below_upper_bound(point, dim)
    
    def above_lower_bound(self, point, dim):
        return point >= self.bounds[0, dim]
        
    def below_upper_bound(self,point,dim):
        return point <= self.bounds[1, dim]
    
    def flip_bit(self, bit):
        bit = 1-bit # flip between 1 and 0
        return bit
    
    def do_progressive_walk(self, starting_zone, seed = None):
        """
        Simulate a progressive random walk using the implementation from Malan 2014. Given a certain starting zone (as a bit array), the walk will be biased towards reaching the other corner.
        """
        
        walk = np.zeros((self.num_steps, self.dim))  # array to store the walk
        np.random.seed(seed)
        
        # First step of the progressive random walk.
        for i in range(self.dim):
            r = np.random.uniform(0, (self.bounds[1,i] - self.bounds[0,i])/2)
            if starting_zone[i] == 1:
                walk[0, i] = self.bounds[1,i] - r
            else:
                walk[0,i] = self.bounds[0,i] + r
                
        # Generate random dimension and constrain to start on the boundary.
        r_dim = np.random.randint(0, self.dim)
        
        if starting_zone[r_dim] == 1:
            walk[0,r_dim] = self.bounds[1,r_dim]
        else:
            walk[0,r_dim] = self.bounds[0,r_dim]
            
            
        # Simulate the rest of the walk.
        for i in range(1,self.num_steps):
            for j in range(self.dim): 
                # Random step
                r = np.random.uniform(0, self.step_size_array[j])
                if starting_zone[j] == 1:
                    r = - r
                temp = walk[i-1, j] + r
                
                # If walk is out of bounds, set next step on walk to be the mirrored position inside the boundary
                if not self.above_lower_bound(temp, j):
                    excess_distance = np.abs(temp - self.bounds[0,j])
                    temp = bounds[0,j] + excess_distance
                    starting_zone[j] = self.flip_bit(starting_zone[j])
                elif not self.below_upper_bound(temp, j):
                    excess_distance = np.abs(temp-self.bounds[1,j])
                    temp = bounds[1,j] - excess_distance
                    starting_zone[j] = self.flip_bit(starting_zone[j])
                    
                # Save this step of the walk.
                walk[i,j] = temp
         

        return walk
    
    def generate_neighbours_for_step(self, point, neighbourhood_size):
        
        neighbours = np.zeros((neighbourhood_size, self.dim))
        
        for i in range(neighbourhood_size):
            for j in range(self.dim):
                r = np.random.uniform(-self.step_size_array[j], self.step_size_array[j])
                temp = point[j] + r
                
                # If outside bounds, use reflection method to get point.
                if not self.above_lower_bound(temp, j):
                    excess_distance = np.abs(temp - self.bounds[0,j])
                    temp = bounds[0,j] + excess_distance
                elif not self.below_upper_bound(temp, j):
                    excess_distance = np.abs(temp-self.bounds[1,j])
                    temp = bounds[1,j] - excess_distance
                
                neighbours[i, j] = temp
        
        return np.atleast_2d(neighbours)
    
    def generate_neighbours_for_walk(self, walk, neighbourhood_size):
        num_points = walk.shape[0]
        dim = self.dim

        # Initialize the array to store neighbors
        neighbours = np.zeros((neighbourhood_size * num_points, dim))

        for i in range(num_points):
            current_neighbours = self.generate_neighbours_for_step(walk[i, :], neighbourhood_size)
            start_index = i * neighbourhood_size
            end_index = (i + 1) * neighbourhood_size
            neighbours[start_index:end_index, :] = current_neighbours

        return neighbours

        
    def check_step_size_prop(self):
        if self.step_size > 0.2:
            message = "Warning: simple RandomWalk may result in an infinite loop for a step size greater than 0.02. Consider using a progressive RW."
            warnings.warn(message)
    
    def do_simple_walk(self, seed=None):
        """
        Simulate the random walk in each dimension, using an unbiased walk.
        """
        self.check_step_size_prop()
        # Initialisation.
        walk = np.zeros((self.num_steps, self.dim))  # array to store the walk
        x = np.zeros(self.dim)
        np.random.seed(seed)

        # First step of the random walk.
        for i in range(self.dim):
            x[i] = (
                self.bounds[0, i]
                + (self.bounds[1, i] - self.bounds[0, i]) * np.random.random()
            )

        # Save the first step.
        walk[0, :] = x

        # Continue the rest of the walk.
        for j in range(1, self.num_steps):
            curr = np.reshape(walk[j - 1, :], (1, -1))  # get previous step's position
            i = 0  # start from first dimension

            while i < self.dim:
                sign = self.random_pm()  # Determine positive or negative direction

                # Defines range of step sizes based on neighbourhood_size
                r = (
                    (self.bounds[0, i] * self.step_size)
                    + ((self.bounds[1, i] - self.bounds[0, i]) * self.step_size)
                ) * np.random.random()
                temp = curr[0, i] + r * sign

                # Handling if the walk leaves the bounds.
                if self.within_bounds(temp, i):
                    # Leave as is
                    s = temp

                    # Saving and iteration to the next dimension.
                    x[i] = s
                    i += 1
                else:
                    # Try and change direction.
                    temp = curr[0, i] - r * sign

                    # if we can't change direction and stay within the bounds, then try a different perturbation vector.
                    if self.within_bounds(temp, i):
                        # Leave as is
                        s = temp

                        # Saving and iteration to the next dimension.
                        x[i] = s
                        i += 1
                    else:
                        continue

            walk[j, :] = x

        return np.atleast_2d(walk)


if __name__ == "__main__":
    

    # Create a 2x2 grid of subplots
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(2):
        for j in range(2):
            # Define bounds for each subplot
            bounds = np.array([[-100, -100], [100, 200]])
            rw = RandomWalk(
                bounds,
                50,
                0.02,
            )

            # Starting zone binary array.
            starting_zone = np.array([i, j])
            
            # Simulate progressive walk.
            walk = rw.do_progressive_walk(starting_zone=starting_zone, seed=None)
            
            # Generate neighbours for each step on the walk.
            neighbours = rw.generate_neighbours_for_walk(walk, neighbourhood_size=3)

            # Plot the random walk on the current subplot
            ax[i, j].plot(walk[:, 0], walk[:, 1], 'b-')  # Thin blue lines connecting points
            # ax[i, j].plot(walk[:, 0], walk[:, 1], 'ro', markersize=3)  # Blue dots at each point
            ax[i, j].plot(neighbours[:, 0], neighbours[:, 1], 'go', markersize=2)  # Green dots at each neighbour
            
            
            # Plot the neighbours

            # Add dotted lines at the bounds on the current subplot
            ax[i, j].axvline(
                x=bounds[0, 0], color="gray", linestyle="--", label="Lower X Bound"
            )
            ax[i, j].axvline(
                x=bounds[1, 0], color="gray", linestyle="--", label="Upper X Bound"
            )
            ax[i, j].axhline(
                y=bounds[0, 1], color="gray", linestyle="--", label="Lower Y Bound"
            )
            ax[i, j].axhline(
                y=bounds[1, 1], color="gray", linestyle="--", label="Upper Y Bound"
            )

    # Adjust subplot spacing
    plt.tight_layout()

    # Show the plot
    plt.show()

    
    
    
    """
    Simple Random Walk.
    """
    # # Create a 5x4 grid of subplots
    # fig, ax = plt.subplots(5, 4, figsize=(10, 10))
    # for i in range(5):
    #     for j in range(4):
    #         # Define bounds for each subplot
    #         bounds = np.array([[-1, 0], [1, 3]])
    #         rw = RandomWalk(
    #             bounds,
    #             1000,
    #             0.2,
    #         )
    #         walk = rw.do_simple_walk(seed=None)

    #         # Plot the random walk on the current subplot
    #         ax[i, j].plot(walk[:, 0], walk[:, 1])

    #         # Add dotted lines at the bounds on the current subplot
    #         ax[i, j].axvline(
    #             x=bounds[0, 0], color="gray", linestyle="--", label="Lower X Bound"
    #         )
    #         ax[i, j].axvline(
    #             x=bounds[1, 0], color="gray", linestyle="--", label="Upper X Bound"
    #         )
    #         ax[i, j].axhline(
    #             y=bounds[0, 1], color="gray", linestyle="--", label="Lower Y Bound"
    #         )
    #         ax[i, j].axhline(
    #             y=bounds[1, 1], color="gray", linestyle="--", label="Upper Y Bound"
    #         )

    # # Adjust subplot spacing
    # plt.tight_layout()

    # # Show the plot
    # plt.show()

# %%
