from setuptools import find_packages, setup


package_name = "tracking"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="shuai",
    maintainer_email="shuai@example.com",
    description="ROS2 RealSense publisher and segmentation tracking nodes.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "realsense_publisher = tracking.realsense_publisher:main",
            "run_segmentation = tracking.run_segmentation:main",
            "visualization = tracking.visualization:main",
        ],
    },
)
