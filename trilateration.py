from trilateration_types import Point, StraightLine, PointDoubleSolution, Measurement3d
import math
import numpy as np


def compute_intersection(m1, m2, m3):
    R_2 = (m2.x - m1.x) ** 2 + (m2.y - m1.y) ** 2
    R = math.sqrt(R_2)
    root = 2 * (m1.r ** 2 + m2.r ** 2) / R_2 - (m1.r ** 2 - m2.r ** 2) ** 2 / R_2 ** 2 - 1
    m_out = Point
    if root >= 0:  # circles intersect in at least one point
        root = math.sqrt(root)
        xm = 0.5 * (m1.x + m2.x) + (m1.r ** 2 - m2.r ** 2) / (2 * R_2) * (m2.x - m1.x) - 0.5 * root * (m2.y - m1.y)
        ym = 0.5 * (m1.y + m2.y) + (m1.r ** 2 - m2.r ** 2) / (2 * R_2) * (m2.y - m1.y) - 0.5 * root * (m1.x - m2.x)
        xp = 0.5 * (m1.x + m2.x) + (m1.r ** 2 - m2.r ** 2) / (2 * R_2) * (m2.x - m1.x) + 0.5 * root * (m2.y - m1.y)
        yp = 0.5 * (m1.y + m2.y) + (m1.r ** 2 - m2.r ** 2) / (2 * R_2) * (m2.y - m1.y) + 0.5 * root * (m1.x - m2.x)
        l1 = compute_loss_function(m1, m2, m3, xm, ym)
        l2 = compute_loss_function(m1, m2, m3, xp, yp)
        if l1 < l2:
            m_out.x = xm
            m_out.y = ym
            return m_out
        else:
            m_out.x = xp
            m_out.y = yp
            return m_out
    else:  # two cases: one circle is inside or outside of the other
        int_c1 = Point()
        int_c2 = Point()
        if m1.r + m2.r < R:  # circles don't overlap
            l1 = points_to_line(m1.x, m1.y, m2.x, m2.y)
            # intersection between straight line and circle 1
            pd1 = line_circle_intersection(l1.slope, l1.intercept, m1.x, m1.y, m1.r)
            dist1 = dist_two_points(pd1.x1, pd1.y1, m2.x, m2.y)
            dist2 = dist_two_points(pd1.x2, pd1.y2, m2.x, m2.y)
            if dist1 < dist2:
                int_c1.x = pd1.x1
                int_c1.y = pd1.y1
            else:
                int_c1.x = pd1.x2
                int_c1.y = pd1.y2
            # intersection between straight line and circle 2
            pd2 = line_circle_intersection(l1.slope, l1.intercept, m2.x, m2.y, m2.r)
            dist1 = dist_two_points(pd2.x1, pd2.y1, m1.x, m1.y)
            dist2 = dist_two_points(pd2.x2, pd2.y2, m1.x, m1.y)
            if dist1 < dist2:
                int_c2.x = pd2.x1
                int_c2.y = pd2.y1
            else:
                int_c2.x = pd2.x2
                int_c2.y = pd2.y2
            m_out.x = (int_c1.x + int_c2.x) / 2
            m_out.y = (int_c1.y + int_c2.y) / 2
            return m_out
        else:  # circles overlap
            if m1.r < m2.r:
                aux = m1
                m1 = m2
                m2 = aux
            l1 = points_to_line(m1.x, m1.y, m2.x, m2.y)
            # intersection between straight line and circle 1
            pd1 = line_circle_intersection(l1.slope, l1.intercept, m1.x, m1.y, m1.r)
            dist1 = dist_two_points(pd1.x1, pd1.y1, m2.x, m2.y)
            dist2 = dist_two_points(pd1.x2, pd1.y2, m2.x, m2.y)
            if dist1 < dist2:
                int_c1.x = pd1.x1
                int_c1.y = pd1.y1
            else:
                int_c1.x = pd1.x2
                int_c1.y = pd1.y2
            # intersection between straight line and circle 2
            pd2 = line_circle_intersection(l1.slope, l1.intercept, m2.x, m2.y, m2.r)
            dist1 = dist_two_points(pd2.x1, pd2.y1, int_c1.x, int_c1.y)
            dist2 = dist_two_points(pd2.x2, pd2.y2, int_c1.x, int_c1.y)
            if dist1 < dist2:
                int_c2.x = pd2.x1
                int_c2.y = pd2.y1
            else:
                int_c2.x = pd2.x2
                int_c2.y = pd2.y2
            m_out.x = (int_c1.x + int_c2.x) / 2
            m_out.y = (int_c1.y + int_c2.y) / 2
            return m_out


def compute_loss_function(m1, m2, m3, x_est, y_est):
    l1 = math.sqrt((m1.x - x_est) ** 2 + (m1.y - y_est) ** 2) - m1.r
    l1 = l1 ** 2
    l2 = math.sqrt((m2.x - x_est) ** 2 + (m2.y - y_est) ** 2) - m2.r
    l2 = l2 ** 2
    l3 = math.sqrt((m3.x - x_est) ** 2 + (m3.y - y_est) ** 2) - m3.r
    l3 = l3 ** 2
    l_out = l1 + l2 + l3
    return l_out


def points_to_line(x1=0, y1=0, x2=0, y2=0):
    line = StraightLine
    line.slope = (y1 - y2) / (x1 - x2)
    line.intercept = y1 - x1 * line.slope
    return line


def line_circle_intersection(slope=0, intersect=0, x=0, y=0, r=0):
    a = slope ** 2 + 1
    b = 2 * x - 2 * slope * intersect + 2 * slope * y
    c = (intersect - y) ** 2 - r ** 2 + x ** 2
    delta = b ** 2 - 4 * a * c
    p = PointDoubleSolution
    p.x1 = (b - math.sqrt(delta)) / (2 * a)
    p.y1 = slope * p.x1 + intersect
    p.x2 = (b + math.sqrt(delta)) / (2 * a)
    p.y2 = slope * p.x2 + intersect
    return p


def dist_two_points(x1=0, y1=0, x2=0, y2=0):
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance


def optimize(m1, m2, m3, p_init):
    x = p_init.x
    y = p_init.y
    min_val = 1000.0
    x_opt = 0
    y_opt = 0
    x_est = 0
    y_est = 0
    for i in np.arange(0, 90):
        for j in np.arange(0, 90):
            x_est = (x - 4.5) + i * 0.1
            y_est = (y - 4.5) + j * 0.1
            cost = compute_loss_function(m1, m2, m3, x_est, y_est)
            if cost < min_val:
                x_opt = x_est
                y_opt = y_est
                min_val = cost
    p_out = Point()
    p_out.x = x_opt
    p_out.y = y_opt
    return p_out


def project_to_ground(measurements):
    for measurement in measurements:
        if measurement.r < measurement.z:
            measurement.r = 0
        else:
            measurement.r = math.sqrt(measurement.r ** 2 - measurement.z ** 2)


def trilateration(m1, m2, m3):
    # convert to m
    m1.to_meter()
    m2.to_meter()
    m3.to_meter()
    # trilateration
    project_to_ground([m1, m2, m3])
    p_init = compute_intersection(m1, m2, m3)
    p_opt = optimize(m1, m2, m3, p_init)
    # convert back to cm
    p_opt.to_centimeter()
    return p_opt


def trilateration_2d(m1, m2, m3):
    # convert to m
    m1.to_meter()
    m2.to_meter()
    m3.to_meter()
    # trilateration
    p_init = compute_intersection(m1, m2, m3)
    p_opt = optimize(m1, m2, m3, p_init)
    # convert back to cm
    p_opt.to_centimeter()
    return p_opt


if __name__ == '__main__':
    m1 = Measurement3d(-0.10389286, -0.03224948, 0.31187567, 3.99618408)
    m2 = Measurement3d(0.94698936, 0.07916058, 0.39984825, 2.45305664)
    m3 = Measurement3d(1.09163498, 1.00071167, 0.22657341, 2.43429520)
    # project_to_ground([m1, m2, m3])
    # # print(m1.x, m1.y, m1.z, m1.r)
    # p_init = compute_intersection(m1, m2, m3)
    # p_opt = optimize(m1, m2, m3, p_init)
    # print(p_init.x, p_init.y)
    # print(p_opt.x, p_opt.y)
    # c1 = compute_loss_function(m1, m2, m3, 8.005826264499783, 4.761039151568663)
    # c2 = compute_loss_function(m1, m2, m3, 3.5058262644998148, 0.76103915156869473)
    # print(c1)
    # print(c2)
    # 0.463330, -0.143473
    # 3.563330,0.656527
