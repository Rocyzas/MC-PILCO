Training:

conda activate mcpilco-new
python test_mcpilco_pensim.py
Plotting (any time during or after training):

python plot_pensim_results.py
Both commands from the project root (MC-PILCO/).

Some decisions made on this implementation

- MC-PILCO implementation for PenSimPy (fully observable)
- state vector: [X, S, DO2, pH, V, T, P] # 7 variables, indices 0–6
- P is observed at every step (rather than only at batch end)
- P_OBSERVABLE_EVERY_STEP flag (False to hide P at every step)
- CONTROL_DISCHARGE flag (flip to True, and set INPUT_DIM=7)
- reward as concentration
- no openAI Gym
