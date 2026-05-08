# RL Rubik's Cube solver

## Requirements
 
- Python 3.10+
- An Nvidia GPU with CUDA (required for model inference)
- The following Python packages:
 
```bash
pip install eel torch matplotlib
```
 
---
 
## How to Run
 
1. Clone the repository
2. Navigate to the `GUI stuff/` folder
3. Run:
 
```bash
python app.py
```
 
The app window will open automatically.
 
---
 
## File Structure
 
The following structure is required for the app to run correctly:
 
```
GUI stuff/
├── app.py                        main entry point
├── model.py                      network and training code
├── two_by_two_cubes.py           2x2 cube environment
├── batched_cubes.py              3x3 cube environment
├── checkpoints/
│   ├── checkpoint_depth_11.pt    2x2 trained model
│   └── checkpoint_depth_retrain.pt    3x3 trained model
└── web/
    ├── index.html                main app page
    ├── results.html              training results viewer
    ├── cubing.min.js             bundled cube renderer (no install needed)
    ├── charts_2x2/               2x2 solve rate charts (cp01–cp14)
    └── charts_3x3/               3x3 solve rate charts (cp01–cp09 + retrain)
```
 
---
