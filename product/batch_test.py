import torch
import numpy as np
import cube
from batched_cubes import batched_cubes

#action indices
U   = 0;  Up  = 1
D   = 2;  Dp  = 3
R   = 4;  Rp  = 5
L   = 6;  Lp  = 7
F   = 8;  Fp  = 9
B   = 10; Bp  = 11
M   = 12; Mp  = 13
E   = 14; Ep  = 15
S   = 16; Sp  = 17

def apply_sequence_old(c, sequence):
    move_map = {
        U:  c.move_U,  Up: c.move_Up,
        D:  c.move_D,  Dp: c.move_Dp,
        R:  c.move_R,  Rp: c.move_Rp,
        L:  c.move_L,  Lp: c.move_Lp,
        F:  c.move_F,  Fp: c.move_Fp,
        B:  c.move_B,  Bp: c.move_Bp,
        M:  c.move_M,  Mp: c.move_Mp,
        E:  c.move_E,  Ep: c.move_Ep,
        S:  c.move_S,  Sp: c.move_Sp,
    }
    for m in sequence:
        move_map[m]()

def apply_sequence_batched(b, sequence):
    for m in sequence:
        b.batched_move(torch.tensor([m], device="cuda"))

#needed to compare with a state on the old representation
def extract_old_tensors(c):
    positions = torch.tensor( np.array([np.array(p.position, dtype=np.float32) for p in c.state]), dtype=torch.float32, device="cuda")
    orientations = torch.tensor( np.array([p.orientation for p in c.state], dtype=np.int64), dtype=torch.int64, device="cuda") 
    return positions, orientations

def compare(old_cube, batch, label):
    old_pos, old_ori = extract_old_tensors(old_cube)

    pos_match = torch.allclose(old_pos, batch.positions[0])
    ori_match = torch.equal(old_ori, batch.orientations[0].to(torch.int64))
    status = "PASS" if (pos_match and ori_match) else "FAIL"
    print(f"[{status}] {label}")
    if not pos_match:
        print("positions mismatch")
    if not ori_match:
        print("orientations mismatch")
        

corner_flip  = [R, U, U, Rp, Up, R, Up, Rp, Lp, U, U, L, U, Lp, U, L]

print("--- test solved state ---")
old = cube.cube()
batch = batched_cubes(1)
compare(old, batch, "solved state matches")
print(f"is_solved() on fresh batch: {batch.is_solved()[0].item()}")

print("\n--- test solo moves --- ")
move_names = ["U","Up","D","Dp","R","Rp","L","Lp","F","Fp","B","Bp","M","Mp","E","Ep","S","Sp"]
for move, name in enumerate(move_names):
    old = cube.cube()
    batch = batched_cubes(1)
    apply_sequence_old(old, [move])
    apply_sequence_batched(batch, [move])
    compare(old, batch, f"single move {name}")




print("\n--- test a move sequence (corner flip) -- ")
old = cube.cube()
batch = batched_cubes(1)
apply_sequence_old(old, corner_flip)
apply_sequence_batched(batch, corner_flip)
compare(old, batch, "corner_flip")

#the point of 2 is to see that sometimes one cube gets a move-invers pair and becomes true
print("\n--- scramble test ---")
batch = batched_cubes(10)
batch.scramble(2)
print(f"is_solved() after 2 move scramble: {batch.is_solved()}")


print("\n--- test moves done on random cubes in a batch --- ")
batch_size = 200
batch = batched_cubes(batch_size)

for move in range(18):
    cube_N = torch.randint(0, batch_size, (1,)).item()

    #-1 as default means no move applied to other cubes
    actions = torch.full((batch_size,), -1, dtype=torch.long, device="cuda")
    actions[cube_N] = move
    batch.batched_move(actions)

    #fresh reference with just this one move
    ref = cube.cube()
    apply_sequence_old(ref, [move])
    ref_pos, ref_ori = extract_old_tensors(ref)

    pos_match = torch.allclose(ref_pos, batch.positions[cube_N])
    ori_match = torch.equal(ref_ori, batch.orientations[cube_N].to(torch.int64))

    if pos_match and ori_match:
        print(f"[PASS] move {move_names[move]} on cube {cube_N}")
    else:
        print(f"[FAIL] move {move_names[move]} on cube {cube_N} (clash)")

print("\n--- corner flip on a batch --- ")
batch = batched_cubes(10)
cube_N = torch.randint(0, 10, (1,)).item()
print(f"applying corner flip to cube {cube_N}")

for m in corner_flip:
    actions = torch.full((10,), -1, dtype=torch.long, device="cuda")
    actions[cube_N] = m
    batch.batched_move(actions)

#reference with corner flip applied
ref = cube.cube()
apply_sequence_old(ref, corner_flip)
ref_pos, ref_ori = extract_old_tensors(ref)

#solved reference for pos/ori comparison
solved_pos, solved_ori = extract_old_tensors(cube.cube())

pos_match_ref = torch.allclose(ref_pos, batch.positions[cube_N])
ori_match_ref = torch.equal(ref_ori, batch.orientations[cube_N].to(torch.int64))
print(f"vs corner flip reference  — pos-match: {pos_match_ref}  ori_match: {ori_match_ref}")
pos_match_solved = torch.allclose(solved_pos, batch.positions[cube_N])
ori_match_solved = torch.equal(solved_ori, batch.orientations[cube_N].to(torch.int64))
print(f"vs solved reference       — pos-match: {pos_match_solved}  ori-match: {ori_match_solved}")