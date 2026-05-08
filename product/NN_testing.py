import torch
import numpy as np
import matplotlib.pyplot as plt
import model
from model import configure, CubeNet, BatchedCubeEnv, select_action, train, batch_size
import os
from datetime import datetime
import ctypes

ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)

try:

    cube_size = "3x3"
    configure(cube_size)

    curriculum = [
        (1, 10),
        (2, 10),
        (3, 10),
        (4, 10),
        (5, 10),
        (6, 10),
        (7, 10),
        (8, 10),
        (9, 10),
        (10, 10),
       
    ]

    curriculum_lr_decay = 1.0  # LR multiplier on each transfer

    # testing parameters
    test_mcts_weight = 0.4
    test_max_steps = 40
    num_test_trials = 3
    test_episodes_per_trial = 100

    # metrics
    plot_window_size = 50
    device = "cuda"


    def test(policy, scramble_depth, use_mcts=True, mcts_weight=test_mcts_weight, log_fn=print):
        test_env = BatchedCubeEnv(test_episodes_per_trial, scramble_depth)
        label = f"Policy+MCTS({mcts_weight})" if use_mcts else "Policy alone"

        for trial in range(num_test_trials):
            states = test_env.reset()
            solved_any = torch.zeros(test_episodes_per_trial, dtype=torch.bool, device=device)

            for step in range(test_max_steps):
                actions, _, _ = select_action(policy, states, test_env, use_mcts=use_mcts, mcts_weight=mcts_weight)
                next_states, rewards, done = test_env.step(actions)
                solved_any |= done
                states = next_states

            solve_count = solved_any.sum().item()
            log(f"  {label} | depth {scramble_depth} | trial {trial+1}: {solve_count}/{test_episodes_per_trial} = {solve_count}%")


    # create output directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(script_dir, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    log_file = open(os.path.join(run_dir, "results.txt"), "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    def plot_rewards(batch_rewards_log, scramble_depth, num_batches):
        episodes_arr = np.arange(1, len(batch_rewards_log) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(episodes_arr, batch_rewards_log, label="Mean Reward per Batch", alpha=0.6)
        if len(batch_rewards_log) >= plot_window_size:
            moving_avg = np.convolve(batch_rewards_log, np.ones(plot_window_size) / plot_window_size, mode='valid')
            plt.plot(episodes_arr[plot_window_size-1:], moving_avg, label=f"Moving Average (window={plot_window_size})", color='red')
        plt.xlabel("Batch")
        plt.ylabel("Mean Reward")
        plt.title(f"Depth {scramble_depth} — batch size {batch_size} — {num_batches} batches")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(run_dir, f"depth_{scramble_depth}.png"))
        plt.close()

    # initialise policy once using depth=1 env for input size
    _init_env = BatchedCubeEnv(batch_size, 1)
    input_size = _init_env.reset().shape[1]
    policy = CubeNet(input_size=input_size, output_size=model.num_moves).to(device)

    for stage, (scramble_depth, stage_batches) in enumerate(curriculum):
        log(f"\n{'='*50}")
        log(f"CURRICULUM STAGE {stage+1}: depth={scramble_depth}, batches={stage_batches}")
        log(f"{'='*50}")

        if scramble_depth > 1:
            policy.load_state_dict(torch.load(f"checkpoint_depth_{scramble_depth-1}.pt"))
            log(f"Loaded weights from depth {scramble_depth-1}")

        env = BatchedCubeEnv(batch_size, scramble_depth)
        lr_scale = curriculum_lr_decay ** stage
        batch_rewards_log = train(env, policy, num_batches=stage_batches, lr_scale=lr_scale)

        torch.save(policy.state_dict(), f"checkpoint_depth_{scramble_depth}.pt")
        log(f"Saved checkpoint_depth_{scramble_depth}.pt")

        plot_rewards(batch_rewards_log, scramble_depth, stage_batches)

        log(f"\n--- Testing depth {scramble_depth} ---")
        test(policy, scramble_depth, use_mcts=True, mcts_weight=test_mcts_weight, log_fn=log)
        test(policy, scramble_depth, use_mcts=False, log_fn=log)

    policy.load_state_dict(torch.load("checkpoint_depth_4.pt"))
    test(policy, 3, use_mcts=False)

finally:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
