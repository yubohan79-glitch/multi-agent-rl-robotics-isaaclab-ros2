from setuptools import setup

package_name = "rcvrl_motion"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", ["config/motion_drift.yaml"]),
        (f"share/{package_name}/launch", [
            "launch/motion_drift_experiment.launch.py",
            "launch/motion_drift_sim_collection.launch.py",
        ]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="RoboCup VisionRL Maintainer",
    maintainer_email="portfolio@example.com",
    description="ROS2 motion telemetry and drift recorder for sim2real calibration.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "motion_drift_recorder = rcvrl_motion.motion_drift_recorder:main",
            "motion_drift_sim_source = rcvrl_motion.motion_drift_sim_source:main",
        ],
    },
)
