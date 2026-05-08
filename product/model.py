import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import batched_cubes as c3
import two_by_two_cubes as c2
import time

# tunable parameters

# training hyperparameters
batch_size = 256
learning_rate = 1e-4
gradient_clip_max = 1.0
baseline_decay = 0.95
distill_alpha = 10  



# mcts parameters
mcts_start_weight = 0.9
mcts_end_weight = 0.5  
mcts_temperature = 2
mcts_frequency = 1
mcts_lookahead = 5
mcts_num_chunks = 2


# reward shaping
md_improvement_scale = 2.0
base_solve_reward = 100

# metrics
device = "cuda"

# configured stuff
num_moves = None
num_pieces = None
batched_cubes = None
solved_positions = None
solved_orientations = None
lossy_schedule = None
use_lossy_threshold = None



move_axes = None  

# sum of positional and orientational distance from solved state for each cube in batch
def manhattan_distance(pos, ori):
    pos_diff = (pos - solved_positions.unsqueeze(0)).abs().sum(dim=-1)
    ori_diff = (ori != solved_orientations.unsqueeze(0)).any(dim=-1).float()
    return (pos_diff + ori_diff).sum(dim=-1)

# sets globals for the selected cube size, including move count, piece count, and pruning schedule
def configure(cube_size):
    global num_moves, num_pieces, batched_cubes, solved_positions, solved_orientations, pos_lookup, move_axes, lossy_schedule, use_lossy_threshold
    c = c2 if cube_size == "2x2" else c3
    num_moves = 12 if cube_size == "2x2" else 18
    num_pieces = 8 if cube_size == "2x2" else 26
    batched_cubes = c.batched_cubes
    solved_positions = c.solved_positions
    solved_orientations = c.solved_orientations

    # lookup table mapping 3d position to piece index
    pos_lookup = torch.zeros(3, 3, 3, dtype=torch.long, device=device)
    for i, p in enumerate(solved_positions):
        x, y, z = (p + 1).long().tolist()
        pos_lookup[x, y, z] = i

    if cube_size == "2x2":
        move_axes = torch.tensor([
            1, 1, 1, 1,  # U, U', D, D'
            0, 0, 0, 0,  # R, R', L, L'
            2, 2, 2, 2,  # F, F', B, B'
        ], dtype=torch.long, device=device)
        lossy_schedule = [1, 3, 3, 8, 8, 10, 11]
        use_lossy_threshold = 4
    else:
        # 3x3 axes - extend later
        move_axes = torch.tensor([
            1, 1, 1, 1,  # U, U', D, D'
            0, 0, 0, 0,  # R, R', L, L'
            2, 2, 2, 2,  # F, F', B, B'
            0, 0,        # M, M'
            1, 1,        # E, E'
            2, 2,        # S, S'
        ], dtype=torch.long, device=device)
        lossy_schedule = [8, 11, 10, 9, 13]
        use_lossy_threshold = 2

# encodes a batched cube object as a flat tensor of one-hot piece ids and orientations
def cube_to_tensor(batch):
    B = batch.B
    idx = (batch.positions + 1).long()
    piece_ids = pos_lookup[idx[:,:,0], idx[:,:,1], idx[:,:,2]]
    one_hot = F.one_hot(piece_ids, num_classes=num_pieces).float()
    orientations = batch.orientations.float()
    encoding = torch.cat([one_hot, orientations], dim=-1)
    return encoding.view(B, -1)

# same as cube_to_tensor but takes raw position and orientation tensors directly
def cube_to_tensor_from(pos, ori):
    B = pos.shape[0]
    idx = (pos + 1).long()
    piece_ids = pos_lookup[idx[:,:,0], idx[:,:,1], idx[:,:,2]]
    one_hot = F.one_hot(piece_ids, num_classes=num_pieces).float()
    encoding = torch.cat([one_hot, ori.float()], dim=-1)
    return encoding.view(B, -1)

# samples actions from a blend of policy and lookahead probabilities, returns log probs under policy only
def select_action(policy, states, env, use_mcts=True, mcts_weight=0.5):
    logits = policy(states)

   

    policy_probs = F.softmax(logits, dim=-1)

    if use_mcts:
        mcts_scores = env.mcts(policy)
        mcts_probs = F.softmax(mcts_scores / mcts_temperature, dim=-1)
        # blend teacher and policy distributions, sample from combined but log prob under policy
        combined_probs = (1 - mcts_weight) * policy_probs + mcts_weight * mcts_probs
        dist_combined = torch.distributions.Categorical(combined_probs)
        actions = dist_combined.sample()
        dist_policy = torch.distributions.Categorical(policy_probs)
        log_probs = dist_policy.log_prob(actions)

        return actions, log_probs, mcts_probs
    else:
        dist = torch.distributions.Categorical(policy_probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, None


# fully connected policy network, maps state encoding to move logits
class CubeNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return self.fc6(x)

# wraps a batched cube environment with reward logic and the lookahead function
class BatchedCubeEnv:
    def __init__(self, B, scramble_depth):
        self.B = B
        self.scramble_depth = scramble_depth
        self.solve_reward = base_solve_reward * scramble_depth
        self.cubes = batched_cubes(B)
        self.active = torch.ones(B, dtype=torch.bool, device=device)

    def reset(self):
        self.cubes.reset()
        self.cubes.scramble(self.scramble_depth)
        self.active = torch.ones(self.B, dtype=torch.bool, device=device)
        return cube_to_tensor(self.cubes)

    def step(self, actions):
        md_before = self.manhattan_distance()
        self.cubes.batched_move(actions)
        done = self.cubes.is_solved()             
        md_after = self.manhattan_distance()
        rewards = self.compute_rewards(md_before, md_after, done)

        self.active = self.active & ~done

        next_states = cube_to_tensor(self.cubes)
        return next_states, rewards, done

    # solve reward overrides md improvement for cubes that reached the goal state
    def compute_rewards(self, md_before, md_after, done):
        improvement = md_before - md_after
        rewards = improvement * md_improvement_scale
        rewards[done] = self.solve_reward
        return rewards

    def manhattan_distance(self):
        pos_diff = (self.cubes.positions - solved_positions.unsqueeze(0)).abs().sum(dim=-1)
        ori_diff = (self.cubes.orientations != solved_orientations.unsqueeze(0)).any(dim=-1).float()
        return (pos_diff + ori_diff).sum(dim=-1)

    # splits batch into chunks and runs lookahead on each, combining results
    def mcts(self, policy):
        B = self.B
        chunk_size = B // mcts_num_chunks
        output = torch.full((B, num_moves), float('-inf'), device=device)

        for chunk_idx in range(mcts_num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, B)
            chunk_pos = self.cubes.positions[start:end].clone()
            chunk_ori = self.cubes.orientations[start:end].clone()

            chunk_output = self._mcts_chunk(policy, chunk_pos, chunk_ori)
            output[start:end] = chunk_output

           

        return output


    # guided tree search with lossless pruning, lossy pruning, and policy rollouts at leaf nodes
    def _mcts_chunk(self, policy, init_pos, init_ori):
        chunk_b = init_pos.shape[0]
        effective_lookahead = min(mcts_lookahead, self.scramble_depth)

        use_lossy = effective_lookahead > use_lossy_threshold

        # scores indexed by (root_cube, first_action), initialised to -inf
        output = torch.full((chunk_b, num_moves), float('-inf'), device=device)

        pos = init_pos.clone()
        ori = init_ori.clone()

        root_idx = torch.arange(chunk_b, device=device)
        first_action = torch.full((chunk_b,), -1, dtype=torch.long, device=device)
        last_move = torch.full((chunk_b,), -1, dtype=torch.long, device=device)
        last_same_axis = torch.full((chunk_b,), -1, dtype=torch.long, device=device)

        all_moves = torch.arange(num_moves, device=device)



        for depth in range(effective_lookahead):
            current_b = pos.shape[0]
            if current_b == 0:
                break

            candidate_idx = all_moves.unsqueeze(0).expand(current_b, -1)

            if depth > 0:
                # lossless pruning: mask inverse of last move and lower-index same-axis commuting moves
                inv_last = (last_move ^ 1).unsqueeze(1)
                inv_mask = (candidate_idx == inv_last)

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

            new_pos = pos[valid_node_idx].clone()
            new_ori = ori[valid_node_idx].clone()

            temp = batched_cubes(total_valid)
            temp.positions = new_pos
            temp.orientations = new_ori
            temp.batched_move(valid_actions)

            pos = temp.positions
            ori = temp.orientations

            new_root_idx = root_idx[valid_node_idx]
            new_first_action = valid_actions if depth == 0 else first_action[valid_node_idx]

            if depth > 0:
                # track highest same-axis move seen on this path for canonical ordering
                parent_axis = move_axes[last_move[valid_node_idx]]
                current_axis = move_axes[valid_actions]
                same_axis_as_parent = (current_axis == parent_axis)
                new_last_same_axis = torch.where(
                    same_axis_as_parent,
                    torch.maximum(last_same_axis[valid_node_idx], valid_actions),
                    valid_actions
                )
            else:
                new_last_same_axis = valid_actions

            new_last_move = valid_actions

            solved = temp.is_solved()

            # propagate solve score back to the first action that initiated this path
            if solved.any():
                depth_discount = 1.0 - 0.01 * depth
                solve_score = float(self.solve_reward) * depth_discount
                flat_idx = new_root_idx[solved] * num_moves + new_first_action[solved]
                output.view(-1).scatter_reduce_(0, flat_idx,
                    torch.full((solved.sum(),), solve_score, device=device),
                    reduce='amax', include_self=True)

            # lossy pruning: discard highest md leaves per schedule to keep tree size manageable
            if use_lossy and depth < len(lossy_schedule):
                k_discard = lossy_schedule[depth]
                if k_discard > 0:
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
                    solved = solved[keep_mask_in_valid]

            active = ~solved
            pos = pos[active]
            ori = ori[active]
            root_idx = new_root_idx[active]
            first_action = new_first_action[active]
            last_move = new_last_move[active]
            last_same_axis = new_last_same_axis[active]

        del temp, new_pos, new_ori, valid_node_idx, valid_actions
        del node_idx_2d, candidate_idx, invalid, valid
        del new_root_idx, new_first_action, new_last_move, new_last_same_axis
        torch.cuda.empty_cache()

     
        # policy rollout on remaining leaves for steps beyond the tree search depth
        rollout_steps = max(0, self.scramble_depth - effective_lookahead)

        if pos.shape[0] > 0:
            if rollout_steps > 0:
                N = pos.shape[0]
                solved_by_policy = torch.zeros(N, dtype=torch.bool, device=device)
                rollout_pos = pos.clone()
                rollout_ori = ori.clone()

                for _ in range(rollout_steps):
                    leaf_states = cube_to_tensor_from(rollout_pos, rollout_ori)
                    with torch.no_grad():
                        policy_actions = policy(leaf_states).argmax(dim=-1)
                    del leaf_states
                    torch.cuda.empty_cache()

                    temp_rollout = batched_cubes(N)
                    temp_rollout.positions = rollout_pos
                    temp_rollout.orientations = rollout_ori
                    temp_rollout.batched_move(policy_actions)

                    solved_by_policy |= temp_rollout.is_solved()
                    rollout_pos = temp_rollout.positions
                    rollout_ori = temp_rollout.orientations
                    torch.cuda.empty_cache()

                pos_diff = (pos - solved_positions.unsqueeze(0)).abs().sum(dim=-1)
                ori_diff = (ori != solved_orientations.unsqueeze(0)).any(dim=-1).float()
                md = (pos_diff + ori_diff).sum(dim=-1)

                # policy solves get a discounted reward, unsolved leaves scored by negative md
                scores = torch.where(solved_by_policy,
                    torch.tensor(float(self.solve_reward) * 0.9, device=device),
                    -md)
            else:
                pos_diff = (pos - solved_positions.unsqueeze(0)).abs().sum(dim=-1)
                ori_diff = (ori != solved_orientations.unsqueeze(0)).any(dim=-1).float()
                md = (pos_diff + ori_diff).sum(dim=-1)
                scores = -md

            # scatter max scores back to their first action at the root
            flat_idx = root_idx * num_moves + first_action
            output.view(-1).scatter_reduce_(0, flat_idx, scores, reduce='amax', include_self=True)

        return output

# reinforce training loop with distillation from lookahead teacher
def train(env, policy, num_batches, lr_scale=1.0):

    optimizer = optim.Adam(policy.parameters(), lr=learning_rate * lr_scale)
    batch_rewards = []
    baseline = 0.0
    max_steps = env.scramble_depth * 2



    for batch in range(num_batches):
        states = env.reset()

        all_log_probs = []
        all_rewards = []
        all_masks = []
        all_states = []
        all_teacher_probs = []

        # teacher weight decays linearly in the second half of training
        progress = batch / num_batches
        if progress < 0.5:
            mcts_weight = mcts_start_weight
        else:
            decay_progress = (progress - 0.5) / 0.5
            mcts_weight = mcts_start_weight - decay_progress * (mcts_start_weight - mcts_end_weight)

        min_steps = env.scramble_depth + 1  
        early_exit_threshold = 0.1  

        for step in range(max_steps):
            use_mcts = (step % mcts_frequency == 0)
            torch.cuda.empty_cache()
            actions, log_probs, teacher_probs = select_action(
                policy, states, env, use_mcts=use_mcts, mcts_weight=mcts_weight
            )

            mask_before = env.active.clone() 
            next_states, rewards, done = env.step(actions)

            all_log_probs.append(log_probs)
            all_rewards.append(rewards)
            all_masks.append(mask_before)   
            if teacher_probs is not None:
                all_states.append(states)
                all_teacher_probs.append(teacher_probs)

            states = next_states

            if step >= min_steps and env.active.sum().item() < env.B * early_exit_threshold:
                break

        all_rewards_tensor = torch.stack(all_rewards)   
        all_masks_tensor   = torch.stack(all_masks)     

        # mask inactive cubes before summing rewards
        masked_rewards = all_rewards_tensor * all_masks_tensor.float()
        G = masked_rewards.sum(dim=0)   



        mean_G = G.mean().item()
        baseline = baseline_decay * baseline + (1 - baseline_decay) * mean_G
        G_normalized = G - baseline

        # reinforce loss: scale log probs by baseline-normalised return, mask inactive cubes
        rl_loss = torch.tensor(0.0, device=device)
        for log_probs, mask in zip(all_log_probs, all_masks):
            step_loss = -(log_probs * mask.float() * G_normalized).mean()
            rl_loss = rl_loss + step_loss
        rl_loss = rl_loss / max_steps

        # distillation loss
        distill_loss = torch.tensor(0.0, device=device)
        if len(all_states) > 0:
            for s, teacher_probs in zip(all_states, all_teacher_probs):
                student_logits = policy(s)
                student_log_probs = F.log_softmax(student_logits, dim=-1)
                step_distill = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
                distill_loss = distill_loss + step_distill
            distill_loss = distill_loss / len(all_states)

        loss = rl_loss + distill_alpha * distill_loss


        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=gradient_clip_max)
        optimizer.step()

        batch_rewards.append(mean_G)

        solve_rate = (torch.stack(all_rewards) >= env.solve_reward).any(dim=0).float().mean().item()
        print(f"Batch {batch+1}: mean reward = {mean_G:.2f}, solve rate = {solve_rate*100:.1f}%")

    return batch_rewards