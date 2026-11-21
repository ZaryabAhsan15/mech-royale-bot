# Mech Royale bot: survivor-v1

Placement scoring rewards surviving, so the bot plays for survival:
- stays inside the closing zone (flow-field navigation around buildings; jet-hops over walls if walled in)
- moves below `sound.step_speed` when calm (no footstep noise) and stands still when safe
- seeks cover (buildings block sight) and regenerates out of sight
- evades visible enemies (away + strafe, facing them to use front armor)
- fires only when attacked, when an enemy is close, or late game; uses laser/lightning/mortar/gun with lead
- picks up nearby weapons and cubes when no enemy is near; takes armor/plating/nanites upgrades

Files: `bot.py` (decisions), `world.py` (map, line of sight, flow field), `mechfight.py` (kit wrapper, unchanged).
`tools/sim.py` is a crude local simulator for catching navigation bugs only.

Libraries: Python standard library only. Kit wrapper `mechfight.py` is from the Builders mechfight-kit.
License: MIT.
