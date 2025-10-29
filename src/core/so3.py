#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-07-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing 3D rotation."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


class SO3:
    """This class represents an SO3 rotations internally represented by rotation
    matrix."""

    def __init__(self, rotation_matrix: ArrayLike | None = None) -> None:
        """Creates a rotation transformation from rot_vector."""
        super().__init__()
        self.rot: np.ndarray = (
            np.asarray(rotation_matrix) if rotation_matrix is not None else np.eye(3)
        )

    @staticmethod
    def exp(rot_vector: ArrayLike) -> SO3:
        """Compute SO3 transformation from a given rotation vector, i.e. exponential
        representation of the rotation."""
        v = np.asarray(rot_vector)
        angle = np.linalg.norm(v)
        omega = v / angle
        SK = np.array([
                [0, -omega[2], omega[1]],
                [omega[2], 0, -omega[0]],
                [-omega[1], omega[0], 0],
            ])
        rot_mat = np.eye(3) + np.sin(angle) * SK + (1 - np.cos(angle)) * (SK @ SK)
        t = SO3(rot_mat)
        return t

    def log(self) -> np.ndarray:
        """Compute rotation vector from this SO3"""
        tr = np.trace(self.rot)
        angle = 0.0
        omega = np.zeros(3)
        
        if tr == -1:
            angle = np.pi
            if self.rot[2,2] != -1:
                omega = (1/np.sqrt(2*(1+self.rot[2,2]))) * np.array([self.rot[0,2], self.rot[1,2], 1+self.rot[2,2]])
            elif self.rot[1,1] != -1:
                omega = (1/np.sqrt(2*(1+self.rot[1,1]))) * np.array([self.rot[0,1], 1+self.rot[1,1], self.rot[2,1]])
            else:
                omega = (1/np.sqrt(2*(1+self.rot[0,0]))) * np.array([1+self.rot[0,0], self.rot[1,0], self.rot[2,0]])
        else:
            angle = np.arccos((tr - 1) / 2)
            if angle != 0:
                omega = (1/(2*np.sin(angle))) * np.array([self.rot[2,1] - self.rot[1,2], self.rot[0,2] - self.rot[2,0], self.rot[1,0] - self.rot[0,1]])
            else:
                omega = np.zeros(3)

        v = np.zeros(3)                
        if angle != 0:
            v = angle * omega

        return v

    def __mul__(self, other: SO3) -> SO3:
        """Compose two rotations, i.e., self * other"""
        new_rot = self.rot @ other.rot
        return SO3(new_rot)

    def inverse(self) -> SO3:
        """Return inverse of the transformation."""
        inverse_rot = self.rot.T.copy()
        return SO3(inverse_rot)

    def act(self, vector: ArrayLike) -> np.ndarray:
        """Rotate given vector by this transformation."""
        v = np.asarray(vector)
        assert v.shape == (3,)
        return self.rot @ v

    def __eq__(self, other: SO3) -> bool:
        """Returns true if two transformations are almost equal."""
        return np.allclose(self.rot, other.rot)

    @staticmethod
    def rx(angle: float) -> SO3:
        """Return rotation matrix around x axis."""
        return SO3(np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ]))

    @staticmethod
    def ry(angle: float) -> SO3:
        """Return rotation matrix around y axis."""
        return SO3(np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ]))


    @staticmethod
    def rz(angle: float) -> SO3:
        """Return rotation matrix around z axis."""
        return SO3(np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ]))

    @staticmethod
    def from_quaternion(q: ArrayLike) -> SO3:
        """Compute rotation from quaternion in a form [qx, qy, qz, qw]."""
        q_xyz = q[:3]
        q_w = q[3]
        norm_q = np.linalg.norm(q)
        so = SO3.exp((2 * np.arccos(q_w / norm_q) / np.linalg.norm(q_xyz)) * q_xyz)
        return so

    def to_quaternion(self) -> np.ndarray:
        """Compute quaternion from self."""
        qv = np.zeros(4)
        rot_vec = self.log()
        omega = rot_vec / np.linalg.norm(rot_vec)
        angle = np.linalg.norm(rot_vec)
        qv[3] = np.cos(angle/2)
        qv[:3] = omega * np.sin(angle/2)
        return qv

    @staticmethod
    def from_angle_axis(angle: float, axis: ArrayLike) -> SO3:
        """Compute rotation from angle axis representation."""
        SK = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ])
        rot_mat = np.eye(3) + np.sin(angle) * SK + (1 - np.cos(angle)) * (SK @ SK)
        return SO3(rot_mat)

    def to_angle_axis(self) -> tuple[float, np.ndarray]:
        """Compute angle axis representation from self."""
        rot_vec = self.log()
        angle = np.linalg.norm(rot_vec)
        axis = rot_vec / angle
        return angle, axis
    
    @staticmethod
    def from_euler_angles(angles: ArrayLike, seq: list[str]) -> SO3:
        """Compute rotation from euler angles defined by a given sequence.
        angles: is a three-dimensional array of angles
        seq: is a list of axis around which angles rotate, e.g. 'xyz', 'xzx', etc.
        """
        # todo: HW1opt: implement from euler angles
        R = SO3()
        for i in range(3):
            if seq[i] == 'x':
                R = R * SO3.rx(angles[i])
            elif seq[i] == 'y':
                R = R * SO3.ry(angles[i])
            elif seq[i] == 'z':
                R = R * SO3.rz(angles[i])
            else:
                raise ValueError("Invalid axis in sequence")
        return R

    def __hash__(self):
        return id(self)
