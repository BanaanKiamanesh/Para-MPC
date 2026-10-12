"""Example plants and problems, defined once and shared by the examples, notebooks, tests and benchmarks.

    quadrotor   planar quadrotor with a slung load: tracking NMPC with hard thrust and soft swing limits
    cartpole    cart-pole swing-up: NMPC with a non-convex cost
    linear      linear plants: the oscillating-masses chain, random LTI systems
"""
from . import cartpole, linear, quadrotor

__all__ = ['cartpole', 'linear', 'quadrotor']
