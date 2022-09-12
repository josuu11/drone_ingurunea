# Parrot Ardrone

Parrot Ardrone on Gazebo

https://bitbucket.org/theconstructcore/parrot_ardrone.git

## How to test

1. Clone the `master` branch
2. `$ roslaunch drone_construct main.launch`
3. `$ rosrun sjtu_drone start_gui`
4. `$ rostopic pub /drone/takeoff std_msgs/Empty "{}"`
5. `$ rosrun image_view image_view image:=/drone/front_camera/image_raw`
6. `$ rosrun teleop_twist_keyboard teleop_twist_keyboard.py`
https://bitbucket.org/theconstructcore/parrot_ardrone.git
