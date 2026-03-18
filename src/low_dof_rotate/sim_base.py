
from leafs import (
    randomizer_mixin,
)
import domain_randomizer

import time

from pydrake.all import (
    System
)

class Simulation:
    def __init__(self):
        # members
        self.diagram = None
        self.simulator = None
        self.systems = []
        self.randomizers: list[randomizer_mixin.RandomizerMixin | domain_randomizer.DomainRandomizer] = []
        
        # plants
        self.plant = None
        self.scene_graph = None
        self.parser = None
        self.builder = None
        
        # analytics
        self.last_wall_time = time.time()
        
    def register_system(self, system):
        # if has the randomizermixin
        if isinstance(system, randomizer_mixin.RandomizerMixin):
            self.randomizers.append(system)
            
        self.systems.append(system)
        
    def create_ports(self):
        pass
    