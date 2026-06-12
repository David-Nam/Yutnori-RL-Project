# Full-Backdo 50M Seed 2 Model

This directory contains the selected final MaskablePPO model trained with:

- ruleset: `full_backdo_v1`
- opponent: `common_rule_based`
- observation: `tactical` (302 values)
- action space: 24
- training seed: 2
- trained timesteps: 50,003,968

Evaluation results:

- official paired evaluation: 61.08% over 5,000 games
- independent confirmation: 60.62% over 5,000 games
- illegal actions: 0
- evaluation errors: 0

The `git_commit` in `config.json` records the repository HEAD at training
start. The full-backdo implementation was still an uncommitted working tree
at that time. The source changes used by this model were subsequently
captured by commits `1ba22f4`, `efef381`, and `6fe1493`.

Use `SHA256SUMS` to verify the release files.
