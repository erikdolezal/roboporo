#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-07-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing 3D transformation."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike

from src.core.so3 import SO3


class SE3:
    """Transformation in 2D that is composed of rotation and translation."""

    def __init__(
        self, translation: ArrayLike | None = None, rotation: SO3 | None = None
    ) -> None:
        """Crete an SE3 transformation. Identity is the default."""
        super().__init__()
        self.translation = (
            np.asarray(translation) if translation is not None else np.zeros(3)
        )
        self.rotation = rotation if rotation is not None else SO3()
        assert self.translation.shape == (3,)

    def __mul__(self, other: SE3) -> SE3:
        """Compose two transformation, i.e., self * other"""
        new_rot = self.rotation.rot @ other.rotation.rot
        new_trans = self.rotation.act(other.translation) + self.translation
        return SE3(new_trans, SO3(new_rot))

    def __add__(self, other: SE3) -> SE3:
        """Translate by another transformation, i.e., self + other"""
        new_trans = self.translation + other.translation
        new_rot = self.rotation * other.rotation
        return SE3(new_trans, new_rot)

    def inverse(self) -> SE3:
        """Compute inverse of the transformation"""
        new_rot = self.rotation.rot.T.copy()
        new_trans = -new_rot @ self.translation
        return SE3(new_trans, SO3(new_rot))

    def act(self, vector: ArrayLike) -> np.ndarray:
        """Rotate given 3D vector by this transformation."""
        v = np.asarray(vector)
        assert v.shape == (3,)
        v = self.rotation.act(v) + self.translation
        return v

    def set_from(self, other: SE3):
        """Copy the properties into current instance."""
        self.translation = other.translation
        self.rotation = other.rotation

    def homogeneous(self) -> np.ndarray:
        """Return homogeneous matrix representation of the transformation."""
        h = np.eye(4)
        h[:3, :3] = self.rotation.rot
        h[:3, 3] = self.translation
        return h
    
    def from_homogeneous(self, H):

        ret = SE3()
        ret.translation = H[:3, 3]
        ret.rotation.rot = H[:3, :3]
        return ret

    def __eq__(self, other: SE3) -> bool:
        """Returns true if two transformations are almost equal."""
        return (
            np.allclose(self.translation, other.translation)
            and self.rotation == other.rotation
        )

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f"(translation={self.translation}, log_rotation={self.rotation.log()})"
