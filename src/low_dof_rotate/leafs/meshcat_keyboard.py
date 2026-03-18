import pydrake.all
from pydrake.all import LeafSystem, AbstractValue

import pydrake.geometry

class MeshcatKeyboardReader(LeafSystem):
    def __init__(self, meshcat: pydrake.geometry.Meshcat):
        LeafSystem.__init__(self)
        self.meshcat = meshcat
        
        # Output: The name of the key pressed (e.g., "KeyW", "ArrowUp")
        self.DeclareAbstractOutputPort(
            "reset_episode",
            alloc=lambda: AbstractValue.Make(0),
            calc=self.CBResetEpisode
        )
        
        self.DeclareAbstractOutputPort(
            "save_episode",
            alloc=lambda: AbstractValue.Make(0),
            calc=self.CBSaveEpisode
        )
        
        # setup buttons
        self.meshcat.AddButton("Reset Episode", "Enter")
        self.meshcat.AddButton("Save Episode", "KeyS")

    def CBResetEpisode(self, context, output):
        output.set_value(self.meshcat.GetButtonClicks("Reset Episode"))
        
    def CBSaveEpisode(self, context, output):
        output.set_value(self.meshcat.GetButtonClicks("Save Episode"))