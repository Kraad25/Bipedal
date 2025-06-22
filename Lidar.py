import Box2D

class LidarCallback(Box2D.b2.rayCastCallback):
    def ReportFixture(self, fixture, point, normal, fraction):
        if (fixture.filterData.categoryBits & 1) == 0:
            return -1
        self.p2 = point
        self.fraction = fraction
        return fraction
