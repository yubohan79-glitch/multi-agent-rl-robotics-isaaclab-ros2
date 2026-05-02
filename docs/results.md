# Results

This file is a submission-ready results template. Replace the placeholders with measured values from the final robot run.

## Test Matrix

| Test | Metric | Result |
| --- | --- | --- |
| AprilTag detection | Tag36h11 ID 1/2/3 recognition | Pending hardware test |
| Single target alignment | Mean alignment time | Pending hardware test |
| Shooter service | Enable/fire/disable success rate | Pending hardware test |
| Nav2 target route | Reached targets / total targets | Pending hardware test |
| Full qualification run | Total time within 180s | Pending hardware test |

## Expected Demonstration Flow

1. Start robot from the yellow start zone.
2. Enable shooter module.
3. Navigate through configured target poses.
4. Search for AprilTag targets.
5. Align by visual feedback.
6. Fire through `/shooter/fire`.
7. Return to the home pose and disable the shooter.

## Evidence To Add

- RViz screenshot showing map, TF and Nav2 path.
- Camera screenshot with detected AprilTag target.
- Terminal screenshot of state transitions.
- Short video of the full 120-second demonstration run.

