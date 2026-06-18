# Software License Agreement (BSD License)
#
# Copyright (c) 2024, The camera_calibration_apriltag authors.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the conditions of the BSD
# license are met.

"""
Hand-eye math for "car-eye" extrinsic calibration of a camera on a mobile base.

A car (``base_link``) drives around in a static world frame (``odom``) while a
rigidly mounted camera observes a static AprilGrid.  This is the classic
eye-in-hand calibration problem ``AX = XB`` with

* ``A`` = motion of the base in odom        (``odom -> base_link``)
* ``B`` = motion of the camera w.r.t. the target (``camera -> grid``)
* ``X`` = the unknown **base_link -> camera** transform we solve for.

OpenCV's :func:`cv2.calibrateHandEye` consumes the per-sample poses directly
(it differences them internally), so we feed it

* ``gripper2base`` = ``odom_T_base_link``     (base pose in odom)
* ``target2cam``   = ``camera_T_grid``        (grid pose in the camera)

and it returns ``cam2gripper`` = ``base_link_T_camera``.

.. note::
   A ground vehicle that only ever drives on a flat floor rotates exclusively
   about the (parallel) vertical axes.  Hand-eye calibration is then degenerate:
   the camera height and the rotation about the vertical axis are weakly
   observable.  Drive with as much variety as the platform allows (ramps,
   pitch/roll, tilting the target) and watch the residual reported by
   :func:`compute_residual`.
"""

import cv2
import numpy as np
import transforms3d as tfs


# OpenCV hand-eye estimators.  Park/Tsai are the classic choices; Daniilidis
# (dual-quaternion) often does best when rotations are small.
AVAILABLE_ALGORITHMS = {
    'Tsai-Lenz': cv2.CALIB_HAND_EYE_TSAI,
    'Park': cv2.CALIB_HAND_EYE_PARK,
    'Horaud': cv2.CALIB_HAND_EYE_HORAUD,
    'Andreff': cv2.CALIB_HAND_EYE_ANDREFF,
    'Daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
}

# Minimum number of samples for a meaningful solve (each pair contributes one
# relative-motion constraint).
MIN_SAMPLES = 3


def transform_to_Rt(transform):
    """geometry_msgs/Transform -> (3x3 rotation, 3-vector translation)."""
    t = transform.translation
    q = transform.rotation
    R = tfs.quaternions.quat2mat((q.w, q.x, q.y, q.z))
    return R, np.array([t.x, t.y, t.z], dtype=np.float64)


def Rt_to_matrix(R, t):
    """(R, t) -> 4x4 homogeneous matrix."""
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    M[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return M


def invert(M):
    """Inverse of a 4x4 rigid transform."""
    R = M[:3, :3]
    t = M[:3, 3]
    Mi = np.eye(4, dtype=np.float64)
    Mi[:3, :3] = R.T
    Mi[:3, 3] = -R.T @ t
    return Mi


def matrix_to_transform_tuple(M):
    """4x4 -> ((tx,ty,tz), (qx,qy,qz,qw)) for filling a Transform message."""
    t = M[:3, 3]
    qw, qx, qy, qz = tfs.quaternions.mat2quat(M[:3, :3])
    return ((float(t[0]), float(t[1]), float(t[2])),
            (float(qx), float(qy), float(qz), float(qw)))


def solve_hand_eye(samples, algorithm='Park'):
    """
    Solve for ``base_link_T_camera`` from a list of samples.

    :param samples: list of dicts, each with
        ``'g2b'`` = (R, t) for ``odom_T_base_link`` and
        ``'t2c'`` = (R, t) for ``camera_T_grid``.
    :param algorithm: key into :data:`AVAILABLE_ALGORITHMS`.
    :returns: 4x4 ``base_link_T_camera`` transform.
    """
    if len(samples) < MIN_SAMPLES:
        raise ValueError("need at least %d samples, have %d"
                         % (MIN_SAMPLES, len(samples)))
    method = AVAILABLE_ALGORITHMS[algorithm]
    Rg = [s['g2b'][0] for s in samples]
    tg = [s['g2b'][1] for s in samples]
    Rc = [s['t2c'][0] for s in samples]
    tc = [s['t2c'][1] for s in samples]
    R_c2g, t_c2g = cv2.calibrateHandEye(Rg, tg, Rc, tc, method=method)
    return Rt_to_matrix(R_c2g, t_c2g)


def compute_residual(samples, X):
    """
    Self-consistency residual for a candidate ``X = base_link_T_camera``.

    The grid is static in odom, so for every sample the implied target pose
    ``odom_T_grid = odom_T_base @ X @ camera_T_grid`` should be identical.  The
    spread of those poses is an interpretable quality metric.

    :returns: dict with ``translation_rms`` (meters) and ``rotation_rms``
        (degrees) of the recovered target pose across samples.
    """
    positions = []
    quats = []
    for s in samples:
        odom_T_base = Rt_to_matrix(*s['g2b'])
        cam_T_grid = Rt_to_matrix(*s['t2c'])
        odom_T_grid = odom_T_base @ X @ cam_T_grid
        positions.append(odom_T_grid[:3, 3])
        q = tfs.quaternions.mat2quat(odom_T_grid[:3, :3])
        # Fix sign ambiguity so quaternions cluster instead of cancelling.
        if quats and np.dot(q, quats[0]) < 0:
            q = -q
        quats.append(q)

    positions = np.asarray(positions)
    trans_rms = float(np.sqrt(np.mean(np.sum(
        (positions - positions.mean(axis=0)) ** 2, axis=1))))

    quats = np.asarray(quats)
    mean_q = quats.mean(axis=0)
    mean_q /= np.linalg.norm(mean_q)
    # Geodesic angle of each quaternion from the mean orientation.
    dots = np.clip(np.abs(quats @ mean_q), 0.0, 1.0)
    angles = 2.0 * np.arccos(dots)
    rot_rms = float(np.degrees(np.sqrt(np.mean(angles ** 2))))

    return {'translation_rms': trans_rms, 'rotation_rms': rot_rms}
