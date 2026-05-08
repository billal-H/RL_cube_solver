import cube
import numpy as np


test_cube = cube.cube()
print(test_cube.is_solved())

def corner_flip(cube):
    cube.move_R()
    cube.move_U()
    cube.move_U()
    cube.move_Rp()
    cube.move_Up()
    cube.move_R()
    cube.move_Up()
    cube.move_Rp()

    cube.move_Lp()
    cube.move_U()
    cube.move_U()
    cube.move_L()
    cube.move_U()
    cube.move_Lp()
    cube.move_U()
    cube.move_L()


def cross_alg(cube):
    cube.move_F()
    cube.move_R()
    cube.move_U()
    cube.move_Rp()
    cube.move_Up()
    cube.move_Fp()

#testing that corners flip properly when doing an algorithm
def test_corner_flip():
    corner_flip(test_cube)
    test_cube.print_unsolved()
    corner_flip(test_cube)
    test_cube.print_unsolved()
    corner_flip(test_cube)
    print(test_cube.is_solved())

#test_corner_flip()

#testing F move works
def test_move_F():
    cross_alg(test_cube)
    cross_alg(test_cube)
    cross_alg(test_cube)
    test_cube.print_unsolved()
    cross_alg(test_cube)
    cross_alg(test_cube)
    cross_alg(test_cube)
    print(test_cube.is_solved())

#test_move_F()

def edge_swap(cube):
    cube.move_Mp()
    cube.move_Mp()
    cube.move_U()
    cube.move_Mp()
    cube.move_Mp()
    cube.move_U()
    cube.move_U()
    cube.move_M()
    cube.move_M()
    cube.move_U()
    cube.move_Mp()
    cube.move_Mp()


def test_move_M():
    edge_swap(test_cube)
    test_cube.print_unsolved()
    edge_swap(test_cube)
    print(test_cube.is_solved())



#test_move_M()


def edge_flip(cube):
    cube.move_R()
    cube.move_Mp()
    cube.move_R()
    cube.move_U()
    cube.move_Rp()
    cube.move_Up()
    cube.move_Rp()
    cube.move_M()
    cube.move_U()
    cube.move_U()
    cube.move_R()
    cube.move_U()
    cube.move_R()
    cube.move_Up()
    cube.move_R()
    cube.move_R()

def test_edge_flip():
    edge_flip(test_cube)
    test_cube.move_U()
    test_cube.move_U()
    test_cube.print_unsolved()
    test_cube.move_U()
    test_cube.move_U()
    edge_flip(test_cube)
    test_cube.print_unsolved()
    edge_flip(test_cube)
    test_cube.move_U()
    test_cube.move_U()
    test_cube.print_unsolved()
    test_cube.move_U()
    test_cube.move_U()
    edge_flip(test_cube)
    print(test_cube.is_solved())



test_edge_flip()
    

"""
#notation explained: each position in the array correlates to an axis. the number there represents which colour is in the position
#so 0 represents the x colour, 1 represents the y colour, and 2 represents the z colour
#so [0,1,2] is the correct one. an R move does [0,2,1], since it moves the z colour to the y axis, does nothing to the x axis, and moves y colour to z axis
corner = np.array([0, 1, 2])
corner2 = np.array([1, 0, 2])

perm = [2, 1, 0]

corner2 = corner2[perm]
print(corner2)

corner2 = corner2[perm]
print(corner2)
"""