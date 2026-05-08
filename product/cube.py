import numpy as np
import random


# list of rotation matricies
# 90 degree rotations in the XY plane. CW is clockwise, CC is counter-clockwise.
rot_XY_CW = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
rot_XY_CC = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

# 90 degree rotations in the XZ plane (around the y-axis when viewed pointing toward you).
rot_XZ_CW = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
rot_XZ_CC = np.array([[0, 0, 1], [0, 1, 0,], [-1, 0, 0]])

# 90 degree rotations in the YZ plane (around the x-axis when viewed pointing toward you).
rot_YZ_CW = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
rot_YZ_CC = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

#perm arrays for tracking piece orientation
R_orient = [0,2,1]
U_orient = [2,1,0]
F_orient = [1,0,2]

class piece:
    def __init__(self, type, colours, position, orientation):
        self.type = type
        self.colours = colours
        self.position = position
        self.orientation = orientation
        """
        L = (-1, 0, 0)
        R = (1, 0, 0)
        U = (0, 1, 0)
        D = (0, -1, 0)
        F = (0, 0, 1)
        B = (0, 0, -1)

        """
    def __repr__(self):
        return f"{self.type}({self.colours},{self.position}, {self.orientation})"
    
    def __eq__(self, other):
        return (
            self.type == other.type and
            self.colours == other.colours and
            np.array_equal(self.position, other.position) and
            np.array_equal(self.orientation, other.orientation)
        )

class cube:
    def solved_state(self):
        return [
            
            # centres, top to bottom
            piece("centre", ["W"], np.array((0, 1, 0)), np.array([0, 0, 0])),
            piece("centre", ["Y"], np.array((0, -1, 0)), np.array([0, 0, 0])),
            # centres, front to back
            piece("centre", ["B"], np.array((0, 0, 1)), np.array([0, 0, 0])),
            piece("centre", ["G"], np.array((0, 0, -1)), np.array([0, 0, 0])),
            # centres, left to right
            piece("centre", ["R"], np.array((-1, 0, 0)), np.array([0, 0, 0])),
            piece("centre", ["O"], np.array((1, 0, 0)), np.array([0, 0, 0])),

            # edges, top
            #for the orientation, -1 means theres no colour in that axis. so the WB edge for example just has a y and z colour
            piece("edge", ["W", "B"], np.array((0, 1, 1)), np.array([-1, 1, 2])), 
            piece("edge", ["W", "R"], np.array((-1, 1, 0)), np.array([0, 1,-1])),
            piece("edge", ["W", "O"], np.array((1, 1, 0)), np.array([0, 1, -1])),
            piece("edge", ["W", "G"], np.array((0, 1, -1)), np.array([-1, 1, 2])),
            # edges, middle
            piece("edge", ["B", "R"], np.array((-1, 0, 1)), np.array([0, -1, 2])),
            piece("edge", ["B", "O"], np.array((1, 0, 1)), np.array([0, -1, 2])),
            piece("edge", ["R", "G"], np.array((-1, 0, -1)), np.array([0, -1, 2])),
            piece("edge", ["O", "G"], np.array((1, 0, -1)), np.array([0, -1, 2])),
            # edges, bottom
            piece("edge", ["B", "Y"], np.array((0, -1, 1)), np.array([-1, 1, 2])),
            piece("edge", ["R", "Y"], np.array((-1, -1, 0)), np.array([0, 1, -1])),
            piece("edge", ["O", "Y"], np.array((1, -1, 0)), np.array([0, 1, -1])),
            piece("edge", ["G", "Y"], np.array((0, -1, -1)), np.array([-1, 1, 2])),

            # corners, top
            piece("corner", ["R", "W", "B"], np.array((-1, 1, 1)), np.array([0, 1, 2])),
            piece("corner", ["O", "W", "B"], np.array((1, 1, 1)), np.array([0, 1, 2])),
            piece("corner", ["R", "W", "G"], np.array((-1, 1, -1)), np.array([0, 1, 2])),
            piece("corner", ["O", "W", "G"], np.array((1, 1, -1)), np.array([0, 1, 2])),
            # corners, bottom
            piece("corner", ["R", "Y", "B"], np.array((-1, -1, 1)), np.array([0, 1, 2])),
            piece("corner", ["O", "Y", "B"], np.array((1, -1, 1)), np.array([0, 1, 2])),
            piece("corner", ["R", "Y", "G"], np.array((-1, -1, -1)), np.array([0, 1, 2])),
            piece("corner", ["O", "Y", "G"], np.array((1, -1, -1)), np.array([0, 1, 2])),
        ]
    def __init__(self, state=None):
            self.state = self.solved_state()

    def is_solved(self):
         #print("cube solved:")
         return self.state == self.solved_state()
    
    def print_state(self):
        for piece in self.state:
            pos = tuple(int(x) for x in piece.position)
            print(f"{piece.type.capitalize()} {piece.colours} at {pos}")

    def print_unsolved(self):
        solved = self.solved_state()
        print("---------currently unsolved--------------")
        for piece, solved_piece in zip(self.state, solved):
            pos_match = all(int(a) == int(b) for a, b in zip(piece.position, solved_piece.position))
            orient_match = np.array_equal(piece.orientation, solved_piece.orientation)
            if not (pos_match and orient_match):
                pos = tuple(int(x) for x in piece.position)
                print(f"{piece.type.capitalize()} {piece.colours} at {pos}, orientation {piece.orientation}")
        print("---------currently unsolved--------------")

    def print_rotations(self):
        for piece in self.state:
             print(piece.orientation)
    
    #general move function 
    def move(self, axis, face, orient, rot_mat):
        for piece in self.state:
            if int(piece.position[axis]) == face: 
                piece.orientation = piece.orientation[orient]
                new_pos = piece.position @ rot_mat.T
                piece.position = tuple(new_pos)
        
    #basic moves
    def move_U(self): self.move(1,1, U_orient, rot_XZ_CW)
    def move_Up(self): self.move(1,1, U_orient, rot_XZ_CC)
    def move_D(self): self.move(1,-1, U_orient, rot_XZ_CC)
    def move_Dp(self): self.move(1,-1, U_orient, rot_XZ_CW)
    def move_R(self): self.move(0,1, R_orient, rot_YZ_CW)
    def move_Rp(self): self.move(0,1, R_orient, rot_YZ_CC)
    def move_L(self): self.move(0,-1, R_orient, rot_YZ_CC)
    def move_Lp(self): self.move(0,-1, R_orient, rot_YZ_CW)
    def move_F(self): self.move(2, 1, F_orient, rot_XY_CW)
    def move_Fp(self): self.move(2, 1, F_orient, rot_XY_CC)
    def move_B(self): self.move(2, -1, F_orient, rot_XY_CC)
    def move_Bp(self): self.move(2, -1, F_orient, rot_XY_CW)

    #slice moves
    def move_M(self): self.move(0,0, R_orient, rot_YZ_CC)
    def move_Mp(self): self.move(0,0, R_orient, rot_YZ_CW)
    def move_E(self): self.move(1,0, U_orient, rot_XZ_CC)
    def move_Ep(self): self.move(1,0, U_orient, rot_XZ_CW)
    def move_S(self): self.move(2,0, F_orient, rot_XY_CW)
    def move_Sp(self): self.move(2,0, F_orient, rot_XY_CC)

    
    #scramble function to apply a given number of random moves
    def scramble(self, num_moves=20, move_map=None):
        move_map = move_map or {
            0: self.move_U, 1: self.move_Up,
            2: self.move_D, 3: self.move_Dp,
            4: self.move_R, 5: self.move_Rp,
            6: self.move_L, 7: self.move_Lp,
            8: self.move_F, 9: self.move_Fp,
            10: self.move_B, 11: self.move_Bp,
            12: self.move_M, 13: self.move_Mp,
            14: self.move_E, 15: self.move_Ep,
            16: self.move_S, 17: self.move_Sp
        }

        for _ in range(num_moves):
            move = random.choice(list(move_map.keys()))
            move_map[move]()
