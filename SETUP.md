# Local setup (Apple Silicon)

Use **arm64** Python (e.g. Homebrew `/opt/homebrew/bin/python3`). x86_64 Anaconda breaks MuJoCo.

```bash
cd value-estimation-vpg
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

Smoke test:

```bash
cd examples/mujoco
python mujoco_a2c_repeat_critic.py \
  --task Hopper-v5 --repeat-critic 50 --seed 0 --logger tensorboard
```

Use `--logger tensorboard` to avoid WandB login. Prefer `Hopper-v5` over deprecated `Hopper-v4`.
