"""Audit turn sign: positive joint angle -> which move?"""
import numpy as np
from callosum.envs import _rubik
print("MOVES:", _rubik.MOVES[:6])
print("F permutation (clockwise from outside):", _rubik.MOVE_TABLES["F"][4:13])
print("F' (counter-clockwise):", np.argsort(_rubik.MOVE_TABLES["F"][4:13]))
