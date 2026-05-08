import torch
import random

#rotation matrices, same as normal cube
rot_XY_CW = torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=torch.float32, device="cuda")
rot_XY_CC = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32, device="cuda")
rot_XZ_CW = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=torch.float32, device="cuda")
rot_XZ_CC = torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=torch.float32, device="cuda")
rot_YZ_CW = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device="cuda")
rot_YZ_CC = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32, device="cuda")

#perm arrays for orientation, same as normal cube 
r_orient = torch.tensor([0, 2, 1], dtype=torch.int64, device="cuda")
u_orient = torch.tensor([2, 1, 0], dtype=torch.int64, device="cuda")
f_orient = torch.tensor([1, 0, 2], dtype=torch.int64, device="cuda")


move_table = [
    (1,  1, u_orient, rot_XZ_CW),   #U
    (1,  1, u_orient, rot_XZ_CC),   #Up
    (1, -1, u_orient, rot_XZ_CC),   #D
    (1, -1, u_orient, rot_XZ_CW),   #Dp
    (0,  1, r_orient, rot_YZ_CW),   #R
    (0,  1, r_orient, rot_YZ_CC),   #Rp
    (0, -1, r_orient, rot_YZ_CC),   #L
    (0, -1, r_orient, rot_YZ_CW),   #Lp
    (2,  1, f_orient, rot_XY_CW),   #F
    (2,  1, f_orient, rot_XY_CC),   #Fp
    (2, -1, f_orient, rot_XY_CC),   #B
    (2, -1, f_orient, rot_XY_CW),   #Bp 
    (0,  0, r_orient, rot_YZ_CC),   #M
    (0,  0, r_orient, rot_YZ_CW),   #Mp
    (1,  0, u_orient, rot_XZ_CC),   #E
    (1,  0, u_orient, rot_XZ_CW),   #Ep
    (2,  0, f_orient, rot_XY_CW),   #S
    (2,  0, f_orient, rot_XY_CC),   #Sp
]

#solved state positions defined as tensor
solved_positions = torch.tensor([
    # centres
    [ 0,  1,  0],  #W
    [ 0, -1,  0],  #Y
    [ 0,  0,  1],  #B
    [ 0,  0, -1],  #G
    [-1,  0,  0],  #R
    [ 1,  0,  0],  #O
    # edges top
    [ 0,  1,  1],  #WB
    [-1,  1,  0],  #WR
    [ 1,  1,  0],  #WO
    [ 0,  1, -1],  #WG
    # edges middle
    [-1,  0,  1],  #BR
    [ 1,  0,  1],  #BO
    [-1,  0, -1],  #RG
    [ 1,  0, -1],  #OG
    # edges bottom
    [ 0, -1,  1],  #BY
    [-1, -1,  0],  #RY
    [ 1, -1,  0],  #OY
    [ 0, -1, -1],  #GY
    # corners top
    [-1,  1,  1],  #RWB
    [ 1,  1,  1],  #OWB
    [-1,  1, -1],  #RWG
    [ 1,  1, -1],  #OWG
    # corners bottom
    [-1, -1,  1],  #RYB
    [ 1, -1,  1],  #OYB
    [-1, -1, -1],  #RYG
    [ 1, -1, -1],  #OYG
], dtype=torch.float32, device="cuda")

# solved state orientations as a tensor
solved_orientations = torch.tensor([
    # centres
    [ 0,  0,  0],  #W
    [ 0,  0,  0],  #Y
    [ 0,  0,  0],  #B
    [ 0,  0,  0],  #G
    [ 0,  0,  0],  #R
    [ 0,  0,  0],  #O
    # edges
    [-1,  1,  2],  #WB
    [ 0,  1, -1],  #WR
    [ 0,  1, -1],  #WO
    [-1,  1,  2],  #WG
    [ 0, -1,  2],  #BR
    [ 0, -1,  2],  #BO
    [ 0, -1,  2],  #RG
    [ 0, -1,  2],  #OG
    [-1,  1,  2],  #BY
    [ 0,  1, -1],  #RY
    [ 0,  1, -1],  #OY
    [-1,  1,  2],  #GY
    # corners
    [ 0,  1,  2],  #RWB
    [ 0,  1,  2],  #OWB
    [ 0,  1,  2],  #RWG
    [ 0,  1,  2],  #OWG
    [ 0,  1,  2],  #RYB
    [ 0,  1,  2],  #OYB
    [ 0,  1,  2],  #RYG
    [ 0,  1,  2],  #OYG
], dtype=torch.int8, device="cuda")


class batched_cubes:

    #init all cubes in a solved position
    def __init__(self, B):
        self.B = B
        self.positions = solved_positions.unsqueeze(0).expand(B, -1, -1).clone()
        self.orientations = solved_orientations.unsqueeze(0).expand(B, -1, -1).clone()
    
    #reset can be targeted to only affect specific cubes
    def reset(self, indices=None):
        if indices is None:
            self.positions = solved_positions.unsqueeze(0).expand(self.B, -1, -1).clone()
            self.orientations = solved_orientations.unsqueeze(0).expand(self.B, -1, -1).clone()
        else:
            self.positions[indices] = solved_positions.unsqueeze(0).expand(len(indices), -1, -1)
            self.orientations[indices] = solved_orientations.unsqueeze(0).expand(len(indices), -1, -1)

    #general move func. uses mask to apply move to all cubes in the batch. 
    #because of move table use a function for each move is no longer needed
    def batched_move(self, actions):
        for move, (axis, face, orient, rot_mat) in enumerate(move_table):
            cube_mask = (actions == move)        

            pos_subset = self.positions[cube_mask]      
            ori_subset = self.orientations[cube_mask]   

            piece_mask = (pos_subset[:, :, axis] == face) 

            pos_subset[piece_mask] = pos_subset[piece_mask] @ rot_mat.T
            ori_subset[piece_mask] = ori_subset[piece_mask][:, orient.long()]

            self.positions[cube_mask] = pos_subset
            self.orientations[cube_mask] = ori_subset

    #torch handles giving each cube a random sequence
    def scramble(self, num_moves):
        for _ in range(num_moves):
            actions = torch.randint(0, 18, (self.B,), device="cuda")
            self.batched_move(actions)

    #returns a tensor instead of jsut one bool
    def is_solved(self):
        pos_match = (self.positions == solved_positions.unsqueeze(0)).all(dim=-1).all(dim=-1)
        ori_match = (self.orientations == solved_orientations.unsqueeze(0)).all(dim=-1).all(dim=-1)
        return pos_match & ori_match