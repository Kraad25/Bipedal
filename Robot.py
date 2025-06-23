from data import *
from Box2D.b2 import revoluteJointDef

class Robot():
    def __init__(self, world, np_random):
        self.world = world
        self.hull = None
        self.legs = []
        self.joints = []        
        self.np_random = np_random

    def create(self):
        initial_x, initial_y = self._get_robot_initial_position()

        self.hull = self._create_hull(initial_x, initial_y)
        self._create_legs(initial_x, initial_y)
    
    def _apply_initial_random_force_to_hull(self):
        random_force_in_x = self.np_random.uniform(-INITIAL_RANDOM_FORCE, INITIAL_RANDOM_FORCE)
        self.hull.ApplyForceToCenter((random_force_in_x, 0), True)

    # Getters #
    def get_joints(self):
        return self.joints
    
    def get_legs(self):
        return self.legs
    
    def get_hull(self):
        return self.hull
    
    def get_drawables(self):
        return [self.hull]+self.legs

    # Private Methods #
    def _get_robot_initial_position(self):
        starting_area = TERRAIN_STARTPAD * TERRAIN_STEP
        init_x = starting_area / 2
        init_y = TERRAIN_HEIGHT + 2 * LEG_H

        return init_x, init_y
    
    def _create_hull(self,x ,y):
        hull = self.world.CreateDynamicBody(position=(x, y), fixtures=HULL_FD)
        hull.color1 = (127, 51, 229)
        hull.color2 = (76, 76, 127)
        return hull
    
    def _create_legs(self, initial_x, initial_y):
        for i in [-1, 1]:
            self._create_leg(i, initial_x, initial_y)

    def _create_leg(self, side, initial_x, initial_y):
        upper_leg  = self.world.CreateDynamicBody(
            position=(initial_x, initial_y - LEG_H / 2 - LEG_DOWN),
            angle=(side*0.05),
            fixtures=LEG_FD,
        )
        upper_leg .color1 = (153 - side * 25, 76 - side * 25, 127 - side * 25)
        upper_leg .color2 = (102 - side * 25, 51 - side * 25, 76 - side * 25)
     
        hip_joint = revoluteJointDef(
            bodyA=self.hull,
            bodyB=upper_leg,
            localAnchorA=(0, LEG_DOWN),
            localAnchorB=(0, LEG_H / 2),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=MOTORS_TORQUE,
            motorSpeed=side,  # ±1 to mirror the direction
            lowerAngle=-0.8, # -45.8366 Degrees
            upperAngle=0.3927, # 22.5000526 Degrees
        )

        upper_leg.ground_contact = False

        self.legs.append(upper_leg)
        self.joints.append(self.world.CreateJoint(hip_joint))

        lower_leg = self.world.CreateDynamicBody(
            position=(initial_x, initial_y - LEG_H * 3 / 2 - LEG_DOWN),
            angle=(side*0.05),
            fixtures=LOWER_FD,
        )
        lower_leg.color1 = (153 - side * 25, 76 - side * 25, 127 - side * 25)
        lower_leg.color2 = (102 - side * 25, 51 - side * 25, 76 - side * 25)

        knee_joint = revoluteJointDef(
            bodyA=upper_leg,
            bodyB=lower_leg,
            localAnchorA=(0, -LEG_H / 2),
            localAnchorB=(0, LEG_H / 2),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=MOTORS_TORQUE,
            motorSpeed=side,  # ±1 to mirror the direction
            lowerAngle=-0.785, # -44.977187 Degrees
            upperAngle=-0.1, # -5.72958 Degrees
        )

        lower_leg.ground_contact = False

        self.legs.append(lower_leg)
        self.joints.append(self.world.CreateJoint(knee_joint))