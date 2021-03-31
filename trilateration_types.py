class Measurement:
    x = 0
    y = 0
    r = 0


class Measurement3d:
    x = 0
    y = 0
    z = 0
    r = 0

    def __init__(self, x, y, z, r):
        self.x = x
        self.y = y
        self.z = z
        self.r = r

    def to_meter(self):
        self.x = self.x/100
        self.y = self.y/100
        self.z = self.z/100
        self.r = self.r/100


class Point:
    x = 0
    y = 0

    def __init__(self):
        self.x = 0
        self.y = 0

    def to_centimeter(self):
        self.x = self.x*100
        self.y = self.y*100


class PointDoubleSolution:
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0


class StraightLine:
    slope = 0
    intercept = 0
