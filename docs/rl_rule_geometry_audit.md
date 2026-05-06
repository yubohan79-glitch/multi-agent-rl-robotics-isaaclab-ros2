# Rule Geometry Audit

Verdict: **PASS**

## Target Table

| name | owner | kind | xy | yaw deg | front probe | plane angles | line blocks | center overlaps | visual overlaps |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| T01_NorthMiddle | blue | normal | (0.18, 1.26) | -45.0 | (0.3921, 1.0479) | 45.0/45.0 | - | - | - |
| T02_NorthEast | blue | normal | (1.26, 1.26) | -135.0 | (1.0479, 1.0479) | 45.0/45.0 | - | - | - |
| T03_WestAboveGate | blue | normal | (-1.26, 0.24) | 45.0 | (-1.0479, 0.4521) | 45.0/45.0 | - | - | - |
| T04_WestBelowGate | yellow | normal | (-1.26, -0.24) | -45.0 | (-1.0479, -0.4521) | 45.0/45.0 | - | - | - |
| T05_EastAboveGate | blue | normal | (1.26, 0.24) | 135.0 | (1.0479, 0.4521) | 45.0/45.0 | - | - | - |
| T06_EastBelowGate | yellow | normal | (1.26, -0.24) | -135.0 | (1.0479, -0.4521) | 45.0/45.0 | - | - | - |
| T07_SouthWest | yellow | normal | (-1.26, -1.26) | 45.0 | (-1.0479, -1.0479) | 45.0/45.0 | - | - | - |
| T08_SouthMiddle | yellow | normal | (-0.18, -1.26) | 135.0 | (-0.3921, -1.0479) | 45.0/45.0 | - | - | - |
| BlueBaseTarget | blue | base_blue | (-1.36, 1.36) | -45.0 | (-1.0489, 1.0489) | 45.0/45.0 | armor_2;armor_3 | - | - |
| YellowBaseTarget | yellow | base_yellow | (1.36, -1.36) | 135.0 | (1.0489, -1.0489) | 45.0/45.0 | armor_6;armor_7 | - | - |

## Physics Checks

| check | pass | details |
| --- | ---: | --- |
| pushable_obstacle_rule_reference_positions | True | box_ne=(0.8, 0.8), box_sw=(-0.8, -0.8), size_m=0.3 |
| yellow_full_armor_base_fire_pose_blocked | True | value=None |
| blue_full_armor_base_fire_pose_blocked | True | value=None |
| pushable_box_moves_and_persists | True | blocked=False, displacement_m=0.045, persisted=True, box_half_m=0.15, robot_radius_m=0.2081 |

## Failures

- None
