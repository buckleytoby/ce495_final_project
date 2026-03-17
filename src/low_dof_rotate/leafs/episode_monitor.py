import pydrake.all
from pydrake.all import LeafSystem, AbstractValue

class BooleanKeyCounter:
    def __init__(self) -> None:
        self.count = 0
        self.triggered = False
        
    def update(self, key_count):
        assert(isinstance(key_count, int))
        
        # check if count incremented
        if key_count > self.count:
            self.count += 1
            self.triggered = True
        else:
            self.triggered = False
            
class KeyTrigger(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        
        # Input from the MeshcatKeyboardReader
        self.DeclareAbstractInputPort("keyboard_input", AbstractValue.Make(0))
        
        # Output: A boolean trigger
        self.DeclareVectorOutputPort("trigger", 1, self.CB)
        
        self.reset_key_counter = BooleanKeyCounter()

        
    def CB(self, context, output):
        key_count = self.get_input_port().Eval(context)
        
        self.reset_key_counter.update(key_count)
        
        output.SetAtIndex(0, self.reset_key_counter.triggered * 1.0)
        
    def upstream_wire(self, builder, keyboard_input_port):
        builder.Connect(
            keyboard_input_port,
            self.get_input_port()
        )

# alias
class ResetTrigger(KeyTrigger):
    pass
       
# alias 
class SaveTrigger(KeyTrigger):
    pass