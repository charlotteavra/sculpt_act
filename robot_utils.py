import time
import numpy as np
from scipy.spatial.transform import Rotation


def goto_grasp(robot, x, y, z, rx, ry, rz, d):
    """
    Parameterize a grasp action by the position [x,y,z] Euler angle rotation [rx,ry,rz], and width [d] of the gripper.
    This function was designed to be used for clay moulding, but in practice can be applied to any task.

    :param fa:  franka robot class instantiation
    """
    pose = robot.get_pose()
    starting_rot = pose.rotation
    orig = Rotation.from_matrix(starting_rot)
    orig_euler = orig.as_euler("xyz", degrees=True)
    rot_vec = np.array([rx, ry, rz])
    new_euler = orig_euler + rot_vec
    r = Rotation.from_euler("xyz", new_euler, degrees=True)
    pose.rotation = r.as_matrix()
    pose.translation = np.array([x, y, z])

    robot.goto_pose(pose)
    robot.goto_gripper(d, force=60.0)
    time.sleep(3)


def get_real_action_from_normalized(action_normalized):
    """ """
    # value = (max - min)*(normalized) + min
    x = (0.63 - 0.55) * action_normalized[0] + 0.55
    y = (0.035 - (-0.035)) * action_normalized[1] + (-0.035)
    # z = (0.2 - 0.178) * action_normalized[2] + 0.178
    z = (0.25 - 0.19) * action_normalized[2] + 0.19
    rz = (90 - (-90)) * action_normalized[3] + (-90)
    d = (0.05 - 0.005) * action_normalized[4] + 0.005

    # clip values to avoid central mount
    if (
        x >= 0.586
        and x <= 0.608
        and y >= -0.011
        and y <= 0.011
        and z >= 0.173
        and z <= 0.181
    ):
        if x <= (0.586 + 0.608) / 2.0:
            x = 0.584
        elif x > (0.586 + 0.608) / 2.0:
            x = 0.61
        elif y <= 0:
            y = -0.013
        elif y > 0:
            y = 0.013
        elif z <= (0.173 + 0.181) / 2.0:
            z = 0.17
        elif z > (0.173 + 0.181) / 2.0:
            z = 0.184

    return np.array([x, y, z, rz, d])
