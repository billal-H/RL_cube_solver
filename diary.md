# Diary

## Friday
Finished report, submitted everything

## Thursday 9/4/2026
Added charts to GUI. More report writing

## Tuesday 7/4/2026
Started report wrting, and GUI. 

## Monday 6/4/2026
Changed mcts method alot, it now does policy rollouts when the lookahead depth is less than the scramble depth.

## Tuesday 24/3/2026
Started experimenting with a 2x2. adding pruning to mcts method.

## Friday 20/3/2026
Representation collapse is unavoidable. Gonna ditch the GNN. 

## Friday 13/3/2026
Star topology GNN implemented. It is worse than before. Gonna try another topology.

## Tuesday 10/3/2026
Some code refactored, curriculum added. considering GNN

## Tuesday 3/3/2026
Batching rewrite done, including tree search changed to use tensor expansion. Some code refactored. 

## Friday 27/2/2026
Decided on a batched approach for speeding up training. Can also add multiprocessing later.

## Friday 20/2/2026
Finally seeing results significantly better than random on small scrambles, demonstrating actual learning of the policy. 
Training still takes forever, so I need to speed that up before I tackle medium sized scrambles.

## Tuesday 17/2/2026
Rewrote a large part of my code to have tensors underlying the operations, hoping that would increase efficiency, but the CPU calls made GPU overhead so bad that it was even slower.
Also tried a simple form of batching, but this had little influence on training speed. Putting this to the side for now, might return later once I actually see success on smaller scrambles. 

## Tuesday 10/2/2026
Bugfixing and tuning parameters. Considering trying to refactor so it will use my GPU more efficiently because training takes a long time. 

## Friday 6/2/2026
Made reward function use manhattan distance. Still doesn't really have good performance. There are alot of parameters I need to tune. 

## Sunday 1/2/2026
Mcts now influences by using a weight value to change how much it influences policy output. Still doesn't actually solve well. 

## Friday 30/1/2026
Started working on adding mcts to action policy. Still unsure of how I want it to influence what actions are chosen.

## Friday 5/12/2025
Made and submitted presentation.

## Wednesday 26/11/2025
Started writing report.

## Thursday 20/11/2025
Improved the training loop and did some testing on it. The skeleton of reinforcement learning should be done now, though the actuall results are still bad because of how simplistic it is
right now. reward function for example just counts solved pieces.

## Wednesday 19/11/2025
Started writing the training function. Does not work yet, probably because of how actions are selected. 

## Sunday 16/11/2025
Network moved out of environment for now. Environment can take cube move functions and track rewards.

## Wednesday 12/11/2025
Started making an environment class for reinforcement learning. Not done, still somewhat unclear on what goes in it. 

## Sunday 9/11/2025
Experimenting with various libraries to begin implementing a neural network

## Sunday 26/10/2025
Added all the basic moves, as well as slice moves. Tested using corner and edge flip algorithms to show that position and orientation tracking work properly.

## Thursday 23/10/2025
The previously described convention for keeping track of orientation was lacking, so I have switched to a different one that represents the orientation of each piece as a vector. Some moves have been implemented with it and work properly. 

## Monday 20/10/2025
Began to define a convention for piece orientation, using the U/D axis as the reference frame. have not fully implemented it yet.

## Wednesday 15/10/2025
Added a .gitignore file. Began implementation of my own cube framework, and tested applying a basic move. The move currently affects more pieces than it should, need to add a face check. I'm still not sure how I want to handle tracking the orientation of pieces, so I have not added that yet.

## Monday 14/10/2025
Experimented with library implementations of rubiks cube, but it proved tedious. Considering building my own representation.

## thursday 9/10/2025
Submitted project plan

## monday 6/10/2025
Cloned the repository onto my machine to prepare to start coding

## thursday 2/10/2025
Had my first meeting with my supervisor. I explained the general concept of my project, and we talked about scope, possible frameworks, etc.

## wednesday 1/10/2025
Started reading research papers about how other attempts at this problem were solved. This helped with identifying what research gaps I wanted to fill. 

