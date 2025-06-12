import Box2D
from Box2D.b2 import contactListener
class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        bodyA = contact.fixtureA.body
        bodyB = contact.fixtureB.body

        # Hull contact triggers game over
        if self.env.hull in [bodyA, bodyB]:
            self.env.game_over = True

        # Check if lower leg (foot) contacts ground
        for i in [1, 3]:  # lower legs
            if self.env.legs[i] in [bodyA, bodyB]:
                self.env.legs[i].ground_contact = True

        # Check if upper leg (knee) contacts ground
        for i in [0, 2]:  # upper legs
            if self.env.legs[i] in [bodyA, bodyB]:
                print(f"Upper leg {i} contacted ground!")
                self.env.game_over = True

    def EndContact(self, contact):
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = False