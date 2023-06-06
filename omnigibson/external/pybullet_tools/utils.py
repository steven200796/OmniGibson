"""
Developed by Caelen Garrett in pybullet-planning repository (https://github.com/caelan/pybullet-planning)
and adapted by iGibson team.
"""
from __future__ import print_function

import colorsys
import json
import math
import os
import pickle
import platform
import numpy as np
import random
import sys
import time
import datetime
from collections import defaultdict, deque, namedtuple
from itertools import product, combinations, count

INF = np.inf
PI = np.pi
BASE_LINK = -1
MAX_DISTANCE = 0.
CLIENT = 0

p = None

def wrap_angle(theta, lower=-np.pi):  # [-np.pi, np.pi)
    return (theta - lower) % (2 * np.pi) + lower


def circular_difference(theta2, theta1):
    return wrap_angle(theta2 - theta1)

def get_num_joints(body):
    return p.getNumJoints(body, physicsClientId=CLIENT)


def get_joints(body):
    return list(range(get_num_joints(body)))


#####################################

# Links

get_links = get_joints  # Does not include BASE_LINK


def get_all_links(body):
    return [BASE_LINK] + list(get_links(body))

#####################################

# Bounding box


AABB = namedtuple('AABB', ['lower', 'upper'])


def aabb_from_points(points):
    return AABB(np.min(points, axis=0), np.max(points, axis=0))


def aabb_union(aabbs):
    return aabb_from_points(np.vstack([aabb for aabb in aabbs]))


def get_aabbs(body):
    return [get_aabb(body, link=link) for link in get_all_links(body)]


def get_aabb(body, link=None):
    # Note that the query is conservative and may return additional objects that don't have actual AABB overlap.
    # This happens because the acceleration structures have some heuristic that enlarges the AABBs a bit
    # (extra margin and extruded along the velocity vector).
    # Contact points with distance exceeding this threshold are not processed by the LCP solver.
    # AABBs are extended by this number. Defaults to 0.02 in Bullet 2.x
    #p.setPhysicsEngineParameter(contactBreakingThreshold=0.0, physicsClientId=CLIENT)
    if link is None:
        aabb = aabb_union(get_aabbs(body))
    else:
        aabb = p.getAABB(body, linkIndex=link, physicsClientId=CLIENT)
    return aabb

def expand_links(body):
    body, links = body if isinstance(body, tuple) else (body, None)
    if links is None:
        links = get_all_links(body)
    return body, links


def any_link_pair_collision(body1, links1, body2, links2=None, **kwargs):
    # TODO: this likely isn't needed anymore
    if links1 is None:
        links1 = get_all_links(body1)
    if links2 is None:
        links2 = get_all_links(body2)
    for link1, link2 in product(links1, links2):
        if (body1 == body2) and (link1 == link2):
            continue
        if pairwise_link_collision(body1, link1, body2, link2, **kwargs):
            #print('body {} link {} body {} link {}'.format(body1, get_link_name(body1, link1), body2, get_link_name(body2, link2)))
            return True
    return False


def body_collision(body1, body2, max_distance=MAX_DISTANCE):  # 10000
    # TODO: confirm that this doesn't just check the base link

    #for i in range(p.getNumJoints(body1)):
    #    for j in range(p.getNumJoints(body2)):
    #        #if len(p.getContactPoints(body1, body2, i, j)) > 0:
    #            #print('body {} {} collide with body {} {}'.format(body1, i, body2, j))

    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance,
                                  physicsClientId=CLIENT)) != 0  # getContactPoints`


def pairwise_collision(body1, body2, **kwargs):
    if isinstance(body1, tuple) or isinstance(body2, tuple):
        body1, links1 = expand_links(body1)
        body2, links2 = expand_links(body2)
        return any_link_pair_collision(body1, links1, body2, links2, **kwargs)
    return body_collision(body1, body2, **kwargs)


def pairwise_link_collision(body1, link1, body2, link2=BASE_LINK, max_distance=MAX_DISTANCE):  # 10000
    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance,
                                  linkIndexA=link1, linkIndexB=link2,
                                  physicsClientId=CLIENT)) != 0  # getContactPoints


#####################################

