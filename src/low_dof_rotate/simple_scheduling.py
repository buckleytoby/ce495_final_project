
"""
simple classes for scheduling exploration rates
"""

import numpy as np


from scipy.interpolate import (
    CubicSpline,
)

import matplotlib.pyplot as plt

class SimpleScheduler:
    def __init__(self,
                 min_t = 0.0,
                 max_t = 100000.0,
                 ):
        self.min_t = min_t
        self.max_t = max_t
        
        # self.create_distribution()
            
    def plot(self):
        # simple plot for debugging
        ts = np.linspace(self.min_t, self.max_t, 100)
        values = self.get_value(ts)
        plt.plot(ts, values)
        plt.show()
        
    def get_value(self, t):
        raise NotImplementedError("get_value method must be implemented by subclasses")
    
    def create_distribution(self):
        """
        interpret the f(t) values as unnormalized probabilities, and sample it.
        """
        num_steps = 100
        
        # 1. Create a grid of points (the 'bins')
        t_grid = np.linspace(self.min_t, self.max_t, num_steps)
        
        # 2. Evaluate the function at these points
        probabilities = self.get_value(t_grid)
        
        # 3. Ensure all values are non-negative and sum to 1 (Normalize)
        probabilities = np.maximum(probabilities, 0)
        probabilities /= probabilities.sum()
        
        self.t_grid = t_grid
        self.probabilities = probabilities
    
    def sample_as_distribution(self):
        # 4. Use np.random.choice to sample an index based on the probabilities
        sampled_index = np.random.choice(len(self.t_grid), p=self.probabilities)
        
        return self.t_grid[sampled_index]

class SimpleExponentialScheduler(SimpleScheduler):
    """
    - initial_value: self explanatory
    - decay_rate: higher is slower decay. 
        - 0.9 -> roughly 43 steps for 1% weight
        - 0.99 -> roughly 460 steps for 1% weight
        - 0.999 -> roughly 4600 steps for 1% weight
        - 0.9999 -> roughly 46000 steps for 1% weight
        - 0.99999 -> roughly 460000 steps for 1% weight
    """
    def __init__(self, initial_value, decay_rate):
        super().__init__()
        
        self.initial_value = initial_value
        self.decay_rate = decay_rate
        
        if False:
            self.plot()

    def get_value(self, t):
        return self.initial_value * (self.decay_rate ** t)
    
class SimpleLogarithmicScheduler(SimpleScheduler):
    def __init__(self, initial_value, decay_rate):
        super().__init__()
        
        self.initial_value = initial_value
        self.decay_rate = decay_rate
        
        if False:
            self.plot()

    def get_value(self, t):
        return self.initial_value / (1 + self.decay_rate * np.log(1 + t))
    
class SimpleConstantScheduler(SimpleScheduler):
    def __init__(self, value):
        super().__init__()
        
        self.value = value
        
        if False:
            self.plot()

    def get_value(self, t):
        return self.value * np.ones_like(t)
    
class SimpleSigmoidLowToHighScheduler(SimpleScheduler):
    def __init__(self, scale = 1.0, max_steps = 100000.0):
        super().__init__(max_t=max_steps)
        
        self.scale = scale
        self.max_steps = max_steps
        
        self.c2 = self.max_steps / 2
        self.c1 = 2.5 / self.c2 # divide by a larger number to smooth out the sigmoid
        
        if False:
            self.plot()

    def get_value(self, t):
        return self.scale / (1 + np.exp(-1.0 * self.c1 * (t - self.c2)))
    
class SimpleSigmoidHighToLowScheduler(SimpleSigmoidLowToHighScheduler):
    def __init__(self, scale = 1.0, max_steps = 100000):
        super().__init__(scale=scale, max_steps=max_steps)
        
        if False:
            self.plot()
                
    def get_value(self, t):
        return 1.0 - super().get_value(t)
    
class SimpleCubicSplineScheduler(SimpleScheduler):
    def __init__(self, scale, x_pts, y_pts):
        super().__init__(0.0, 1.0)
        self.scale = scale
        self.x_pts = x_pts
        self.y_pts = y_pts
        
        self.spline = CubicSpline(self.x_pts, self.y_pts, bc_type='clamped')
        
        if False:
            self.plot()

    def get_value(self, t):
        return self.scale * self.spline(t)
    
class SimpleSCurveLowToHighScheduler(SimpleCubicSplineScheduler):
    def __init__(self, scale = 1.0):
        x_pts = [0.0, 0.25, 0.5, 0.75, 1.0]
        y_pts = [0.0, 0.1, 0.5, 0.9, 1.0]
        super().__init__(scale=scale, x_pts=x_pts, y_pts=y_pts)
        
class SimpleLinearScheduler(SimpleScheduler):
    """
    go from (xa, ya) to (xb, yb) in a linear way. outside of that, the value is clipped to ya or yb.
    """
    def __init__(self, xa, ya, xb, yb):
        super().__init__(min_t=xa, max_t=xb)
        self.xa = xa
        self.ya = ya
        self.xb = xb
        self.yb = yb
        
        self.slope = (yb - ya) / (xb - xa)
        
        if False:
            self.plot()

    def get_value(self, t):
        y = self.ya + self.slope * (t - self.xa)
        
        y = np.clip(y, min(self.ya, self.yb), max(self.ya, self.yb))
        return y
    
    
############################################################
############################################################

class MultipleSchedulersSampler:
    """
    given multiple schedulers, sample from their summed probabilities
    """
    def __init__(self, schedulers: list[SimpleScheduler]):
        self.schedulers = schedulers

    def sample(self, t):
        values = [scheduler.get_value(t) for scheduler in self.schedulers]
        total = sum(values)
        probabilities = [value / total for value in values]

        # sample from the probabilities
        choice = np.random.choice(len(self.schedulers), p=probabilities)

        return choice