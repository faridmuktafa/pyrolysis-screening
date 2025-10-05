class Actuator:
    def __init__(self): self.state='IDLE'
    def route_target(self): self.state='TARGET'  # TODO: GPIO/PLC control
    def route_reject(self): self.state='REJECT'
    def idle(self): self.state='IDLE'
