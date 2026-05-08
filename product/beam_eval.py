import torch
import batched_cubes as c3
import time

device = "cuda"
num_moves = 18
num_pieces = 26
solved_positions = c3.solved_positions
solved_orientations = c3.solved_orientations

move_axes = torch.tensor([
    1, 1, 1, 1,   # U, U', D, D'
    0, 0, 0, 0,   # R, R', L, L'
    2, 2, 2, 2,   # F, F', B, B'
    0, 0,         # M, M'
    1, 1,         # E, E'
    2, 2,         # S, S'
], dtype=torch.long, device=device)

def manhattan_distance(pos, ori):
    pos_diff = (pos - solved_positions.unsqueeze(0)).abs().sum(dim=-1)
    ori_diff = (ori != solved_orientations.unsqueeze(0)).any(dim=-1).float()
    return (pos_diff + ori_diff).sum(dim=-1)

def generate_scrambles(n, scramble_depth):
    cubes = c3.batched_cubes(n)
    cubes.scramble(scramble_depth)
    return cubes.positions.clone(), cubes.orientations.clone()

def tree_search(init_pos, init_ori, max_depth, k_discard_schedule, verbose=False):
    B = init_pos.shape[0]
    all_moves = torch.arange(num_moves, device=device)

    pos = init_pos.clone()
    ori = init_ori.clone()

    root_idx = torch.arange(B, device=device)
    first_action = torch.full((B,), -1, dtype=torch.long, device=device)
    last_move = torch.full((B,), -1, dtype=torch.long, device=device)
    last_same_axis = torch.full((B,), -1, dtype=torch.long, device=device)

    solved_roots = torch.zeros(B, dtype=torch.bool, device=device)
    peak_tree_size = 0

    for depth in range(max_depth):
        current_b = pos.shape[0]
        if current_b == 0:
            break

        if isinstance(k_discard_schedule, list):
            k_discard = k_discard_schedule[depth] if depth < len(k_discard_schedule) else k_discard_schedule[-1]
        else:
            k_discard = k_discard_schedule

        candidate_idx = all_moves.unsqueeze(0).expand(current_b, -1)

        if depth > 0:
            inv_mask = (candidate_idx == (last_move ^ 1).unsqueeze(1))
            last_axis = move_axes[last_move]
            candidate_axes = move_axes.unsqueeze(0).expand(current_b, -1)
            same_axis = (candidate_axes == last_axis.unsqueeze(1))
            lower = (candidate_idx < last_same_axis.unsqueeze(1))
            commute_mask = same_axis & lower
            invalid = inv_mask | commute_mask
        else:
            invalid = torch.zeros(current_b, num_moves, dtype=torch.bool, device=device)

        valid = ~invalid
        node_idx_2d = torch.arange(current_b, device=device).unsqueeze(1).expand(current_b, num_moves)
        valid_node_idx = node_idx_2d[valid]
        valid_actions = candidate_idx[valid]
        total_valid = valid_node_idx.shape[0]
        peak_tree_size = max(peak_tree_size, total_valid)

        new_pos = pos[valid_node_idx].clone()
        new_ori = ori[valid_node_idx].clone()

        temp = c3.batched_cubes(total_valid)
        temp.positions = new_pos
        temp.orientations = new_ori
        temp.batched_move(valid_actions)

        pos = temp.positions
        ori = temp.orientations

        new_root_idx = root_idx[valid_node_idx]
        new_first_action = valid_actions if depth == 0 else first_action[valid_node_idx]

        if depth > 0:
            parent_axis = move_axes[last_move[valid_node_idx]]
            current_axis = move_axes[valid_actions]
            same_axis_as_parent = (current_axis == parent_axis)
            new_last_same_axis = torch.where(
                same_axis_as_parent,
                torch.maximum(last_same_axis[valid_node_idx], valid_actions),
                valid_actions)
        else:
            new_last_same_axis = valid_actions

        new_last_move = valid_actions

        # md lossy pruning
        if k_discard > 0 and total_valid > 0:
            md = manhattan_distance(pos, ori)
            md_full = torch.full((current_b, num_moves), float('inf'), device=device)
            md_full[valid] = md

            valid_counts = valid.sum(dim=-1)
            min_keep = (valid_counts - k_discard).clamp(min=1)
            md_sorted, _ = md_full.sort(dim=-1)
            threshold = md_sorted[torch.arange(current_b, device=device), min_keep - 1]
            prune_by_md = (md_full > threshold.unsqueeze(1)) & valid
            keep = valid & ~prune_by_md
            keep_mask_in_valid = keep[valid]

            pos = pos[keep_mask_in_valid]
            ori = ori[keep_mask_in_valid]
            new_root_idx = new_root_idx[keep_mask_in_valid]
            new_first_action = new_first_action[keep_mask_in_valid]
            new_last_move = new_last_move[keep_mask_in_valid]
            new_last_same_axis = new_last_same_axis[keep_mask_in_valid]

        solved_check = c3.batched_cubes(pos.shape[0])
        solved_check.positions = pos
        solved_check.orientations = ori
        solved = solved_check.is_solved()

        if solved.any():
            solved_roots[new_root_idx[solved]] = True

        active = ~solved
        pos = pos[active]
        ori = ori[active]
        root_idx = new_root_idx[active]
        first_action = new_first_action[active]
        last_move = new_last_move[active]
        last_same_axis = new_last_same_axis[active]

        if depth == max_depth - 1:
            print(f"  leaf count (rollout input size): {pos.shape[0]}")

        if verbose:
            print(f"    ply {depth}: k={k_discard}, expanded={total_valid}, remaining={pos.shape[0]}")

    return solved_roots, peak_tree_size


def run_tests(n_scrambles=256, scramble_depth=4, max_search_depth=4):
    print(f"Generating {n_scrambles} scrambles at depth {scramble_depth}...")
    init_pos, init_ori = generate_scrambles(n_scrambles, scramble_depth)
    torch.cuda.synchronize()

    fixed_ks = [0, 2, 4, 6, 8, 10, 12]

    schedules = {
    # --- accuracy focus: loosen pruning vs balanced_heavy ---
    "acc_1":  [7, 11, 10,  9, 10],  # ease ply0 slightly
    "acc_2":  [8, 10, 10,  9, 10],  # ease ply1 slightly
    "acc_3":  [8, 11,  9,  9, 10],  # ease ply2 slightly
    "acc_4":  [8, 11, 10,  8, 10],  # ease ply3 slightly
    "acc_5":  [8, 11, 10,  9,  9],  # ease ply4 slightly
    "acc_6":  [7, 11, 10,  8, 10],  # ease ply0 and ply3

    # --- tree size focus: tighten pruning vs balanced_heavy ---
    "size_1": [9, 11, 10,  9, 10],  # squeeze ply0 harder
    "size_2": [8, 12, 10,  9, 10],  # squeeze ply1 harder
    "size_3": [8, 11, 11,  9, 10],  # squeeze ply2 harder
    "size_4": [8, 11, 10, 11, 10],  # squeeze ply3 harder
    "size_5": [8, 11, 10,  9, 12],  # squeeze ply4 harder
    "size_6": [9, 11, 11,  9, 10],  # squeeze ply0+ply2
    "size_7": [8, 12, 10, 11, 10],  # squeeze ply1+ply3

     "size_ply4_a": [8, 11, 10,  9, 13],
    "size_ply4_b": [8, 11, 10,  9, 14],
    "size_ply4_c": [8, 11, 10,  9, 15],
    "size_ply4_d": [9, 11, 10,  9, 13],
    "size_ply4_e": [9, 11, 10,  9, 14],
}
    print(f"\n--- Fixed k_discard (depth {max_search_depth}) ---")
    ref_solved = None
    for k in fixed_ks:
        torch.cuda.synchronize()
        try:
            start = time.time()
            solved, peak = tree_search(init_pos, init_ori, max_search_depth, k, verbose=(k==0))
            torch.cuda.synchronize()
            elapsed = time.time() - start
            n_solved = solved.sum().item()
            if k == 0:
                ref_solved = solved
                missed = 0
            else:
                missed = (ref_solved & ~solved).sum().item() if ref_solved is not None else -1
            print(f"  k={k}: solved {n_solved}/{n_scrambles} ({100*n_solved/n_scrambles:.1f}%), "
                  f"missed {missed}, peak_tree={peak:,}, time {elapsed:.2f}s")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  k={k}: OOM")

    print(f"\n--- Variable schedules (depth {max_search_depth}) ---")
    for name, schedule in schedules.items():
        torch.cuda.synchronize()
        try:
            start = time.time()
            solved, peak = tree_search(init_pos, init_ori, max_search_depth, schedule, verbose=False)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            n_solved = solved.sum().item()
            missed = (ref_solved & ~solved).sum().item() if ref_solved is not None else -1
            print(f"  {name} {schedule}: solved {n_solved}/{n_scrambles} ({100*n_solved/n_scrambles:.1f}%), "
                  f"missed {missed}, peak_tree={peak:,}, time {elapsed:.2f}s")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  {name}: OOM")


if __name__ == "__main__":
    run_tests(
        n_scrambles=256,
        scramble_depth=5,
        max_search_depth=5,
    )