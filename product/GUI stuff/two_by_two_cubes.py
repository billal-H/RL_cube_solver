import torch

rot_XY_CW = torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=torch.float32, device="cuda")
rot_XY_CC = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32, device="cuda")
rot_XZ_CW = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=torch.float32, device="cuda")
rot_XZ_CC = torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=torch.float32, device="cuda")
rot_YZ_CW = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device="cuda")
rot_YZ_CC = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32, device="cuda")

r_orient = torch.tensor([0, 2, 1], dtype=torch.int64, device="cuda")
u_orient = torch.tensor([2, 1, 0], dtype=torch.int64, device="cuda")
f_orient = torch.tensor([1, 0, 2], dtype=torch.int64, device="cuda")

# 2x2 has no middle slices — only 12 outer face moves
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
]

# 2x2 has only 8 corners
solved_positions = torch.tensor([
    [-1,  1,  1],  #RWB
    [ 1,  1,  1],  #OWB
    [-1,  1, -1],  #RWG
    [ 1,  1, -1],  #OWG
    [-1, -1,  1],  #RYB
    [ 1, -1,  1],  #OYB
    [-1, -1, -1],  #RYG
    [ 1, -1, -1],  #OYG
], dtype=torch.float32, device="cuda")

solved_orientations = torch.tensor([
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

    def __init__(self, B):
        self.B = B
        self.positions = solved_positions.unsqueeze(0).expand(B, -1, -1).clone()
        self.orientations = solved_orientations.unsqueeze(0).expand(B, -1, -1).clone()

    def reset(self, indices=None):
        if indices is None:
            self.positions = solved_positions.unsqueeze(0).expand(self.B, -1, -1).clone()
            self.orientations = solved_orientations.unsqueeze(0).expand(self.B, -1, -1).clone()
        else:
            self.positions[indices] = solved_positions.unsqueeze(0).expand(len(indices), -1, -1)
            self.orientations[indices] = solved_orientations.unsqueeze(0).expand(len(indices), -1, -1)

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

    def scramble(self, num_moves):
        for _ in range(num_moves):
            actions = torch.randint(0, 12, (self.B,), device="cuda")
            self.batched_move(actions)

    def is_solved(self):
        pos_match = (self.positions == solved_positions.unsqueeze(0)).all(dim=-1).all(dim=-1)
        ori_match = (self.orientations == solved_orientations.unsqueeze(0)).all(dim=-1).all(dim=-1)
        return pos_match & ori_match