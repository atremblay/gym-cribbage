# -*- coding: utf-8 -*-
# @Author: Marc-Antoine
# @Date:   2019-03-17 17:18:42
# @Last Modified by:   Marc-Antoine Belanger
# @Last Modified time: 2019-03-17 17:20:31

from gym.envs.registration import register

register(
    id='cribbage-v0',
    entry_point='gym_cribbage.envs:CribbageEnv',
)
