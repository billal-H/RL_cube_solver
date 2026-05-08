import eel
import torch
import random
import model as m

# move name tables
move_names_2x2 = ["U", "U'", "D", "D'", "R", "R'", "L", "L'", "F", "F'", "B", "B'"]
move_names_3x3 = ["U", "U'", "D", "D'", "R", "R'", "L", "L'", "F", "F'", "B", "B'", "M", "M'", "E", "E'", "S", "S'"]

move_to_idx_2x2 = {mv: i for i, mv in enumerate(move_names_2x2)}
move_to_idx_3x3 = {mv: i for i, mv in enumerate(move_names_3x3)}

# load 2x2 model
m.configure("2x2")
_init_env_2x2 = m.BatchedCubeEnv(1, 1)
input_size_2x2 = _init_env_2x2.reset().shape[1]
policy_2x2 = m.CubeNet(input_size=input_size_2x2, output_size=12).to("cuda")
policy_2x2.load_state_dict(torch.load("checkpoints/checkpoint_depth_11.pt", map_location="cuda"))
policy_2x2.eval()
print("2x2 model loaded")

# load 3x3 model
m.configure("3x3")
_init_env_3x3 = m.BatchedCubeEnv(1, 1)
input_size_3x3 = _init_env_3x3.reset().shape[1]
policy_3x3 = m.CubeNet(input_size=input_size_3x3, output_size=18).to("cuda")
policy_3x3.load_state_dict(torch.load("checkpoints/checkpoint_depth_retrain.pt", map_location="cuda"))
policy_3x3.eval()
print("3x3 model loaded")

# default to 3x3 matching JS default
current_cube_type = "3x3x3"
m.configure("3x3")


@eel.expose
def set_cube_type(cube_type):
    global current_cube_type
    current_cube_type = cube_type
    m.configure("3x3" if cube_type == "3x3x3" else "2x2")
    print(f"Switched to {cube_type}")

temperature = 5
@eel.expose
def solve(move_list):
    is_2x2 = current_cube_type == "2x2x2"

    policy = policy_2x2 if is_2x2 else policy_3x3
    move_to_idx = move_to_idx_2x2 if is_2x2 else move_to_idx_3x3
    move_names = move_names_2x2 if is_2x2 else move_names_3x3

    B = 200
    cube = m.batched_cubes(B)
    for mv in move_list:
        if mv not in move_to_idx:
            print(f"Unknown move: {mv}")
            return None
        action = torch.tensor([move_to_idx[mv]] * B, device="cuda")
        cube.batched_move(action)

    if cube.is_solved().any():
        return ""

    max_steps = 50
    solution_moves = [[] for _ in range(B)]
    _first_step = True

    for _ in range(max_steps):
        state = m.cube_to_tensor(cube)
        with torch.no_grad():
            logits = policy(state)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * probs.log()).sum(dim=-1)

            if entropy.mean().item() < 0.5:
                probs = torch.softmax(logits / temperature, dim=-1)
                if _first_step:
                    new_entropy = -(probs * probs.log()).sum(dim=-1)
                    print(f"low entropy detected, applied temperature={temperature}")
                    print(f"entropy before={entropy.mean().item():.3f} after={new_entropy.mean().item():.3f}")
            else:
                if _first_step:
                    print(f"normal entropy={entropy.mean().item():.3f}, no temperature needed")

            _first_step = False
            actions = torch.distributions.Categorical(probs).sample()

        cube.batched_move(actions)

        for i in range(B):
            solution_moves[i].append(move_names[actions[i].item()])

        solved = cube.is_solved()
        if solved.any():
            idx = solved.nonzero(as_tuple=True)[0][0].item()
            return " ".join(solution_moves[idx])

    return None

@eel.expose
def get_scramble(depth):
    is_2x2 = current_cube_type == "2x2x2"
    move_names = move_names_2x2 if is_2x2 else move_names_3x3
    scramble = []
    last = None
    for _ in range(depth):
        choices = [mv for mv in move_names if mv != last and
                   mv != (last + "'" if last and "'" not in last else (last[:-1] if last else None))]
        mv = random.choice(choices)
        scramble.append(mv)
        last = mv
    return scramble


eel.init("web")
eel.start("index.html", size=(900, 700))