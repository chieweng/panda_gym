import numpy as np

def euler_to_quaternion(Rx: float, Ry: float, Rz: float) -> np.ndarray:
    """
    Convert Euler angles (Rx, Ry, Rz) in radians to a quaternion.

    Args:
        Rx (float): Rotation angle around the X-axis in radians.
        Ry (float): Rotation angle around the Y-axis in radians.
        Rz (float): Rotation angle around the Z-axis in radians.

    Returns:
        np.ndarray: A numpy array representing the quaternion (x, y, z, w).
    """
    x = np.sin(Rx / 2)
    y = np.sin(Ry / 2)
    z = np.sin(Rz / 2)
    w = np.cos(Rx / 2) * np.cos(Ry / 2) * np.cos(Rz / 2) + np.sin(Rx / 2) * np.sin(Ry / 2) * np.sin(Rz / 2)

    return np.array([x, y, z, w])

def rot_matrix_to_quat(R):
        trace = np.trace(R)
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2  # S=4*qw
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:  # Column 1
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*x
                w = (R[2, 1] - R[1, 2]) / S
                x = 0.25 * S
                y = (R[0, 1] + R[1, 0]) / S
                z = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:  # Column 2
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*y
                w = (R[0, 2] - R[2, 0]) / S
                x = (R[0, 1] + R[1, 0]) / S
                y = 0.25 * S
                z = (R[1, 2] + R[2, 1]) / S
            else:  # Column 3
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*z
                w = (R[1, 0] - R[0, 1]) / S
                x = (R[0, 2] + R[2, 0]) / S
                y = (R[1, 2] + R[2, 1]) / S
                z = 0.25 * S
        
        return np.array([x, y, z, w])
