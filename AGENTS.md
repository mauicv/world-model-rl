# AGENTS.md

## Cursor Cloud specific instructions

This is a pure-Python ML/RL research library (`world-model-rl`). No web services, databases, or containers needed.

### Quick reference

| Action | Command |
|--------|---------|
| Install deps | `source venv/bin/activate && pip install -e .` |
| Run tests | `source venv/bin/activate && pytest src` |
| Run example | `source venv/bin/activate && python src/reflect/examples/differential_rl/basic.py` |

### Non-obvious notes

- **`torchvision` is an implicit dependency** not listed in `pyproject.toml` but required at runtime (imported in `reflect.data.loader`). The update script installs it explicitly.
- **2 tests fail on PyTorch >=2.6** (`test_save_and_load`, `test_save_load`) due to the `weights_only=True` default in `torch.load`. This is a pre-existing issue, not an environment problem.
- **No linter is configured** in the repository (no ruff, flake8, pylint, mypy, or pyright config).
- The `EnvDataLoader.sample()` returns a tuple `(b_inds, t_inds, states, actions, rewards, dones)`, not a dict.
- MuJoCo is bundled automatically via `gymnasium[mujoco]`; no separate system install required.
- The venv lives at `/workspace/venv`.
