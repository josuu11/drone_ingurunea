#!/usr/bin/env python
from gym.envs.registration import register
from gym import envs


def RegisterOpenAI_Ros_Env(task_env, max_episode_steps=10000):
    """
    Registers all the ENVS supported in OpenAI ROS. This way we can load them
    with variable limits.
    Here is where you have to PLACE YOUR NEW TASK ENV, to be registered and accesible.
    return: False if the Task_Env wasnt registered, True if it was.
    """

    ###########################################################################
    # MovingCube Task-Robot Envs

    result = True

    # Cubli Moving Cube
    if task_env == 'MovingCubeOneDiskWalk-v0':
        print("Import module")

        # We have to import the Class that we registered so that it can be found afterwards in the Make
        from openai_ros.task_envs.moving_cube import one_disk_walk

        print("Importing register env")
        # We register the Class through the Gym system
        register(
            id=task_env,
            #entry_point='openai_ros:task_envs.moving_cube.one_disk_walk.MovingCubeOneDiskWalkEnv',
            entry_point='openai_ros.task_envs.moving_cube.one_disk_walk:MovingCubeOneDiskWalkEnv',
            max_episode_steps=max_episode_steps,
        )

    elif task_env == 'ParrotDroneGoto-v0':

        register(
            id=task_env,
            entry_point='openai_ros.task_envs.parrotdrone.parrotdrone_goto:ParrotDroneGotoEnv',
            max_episode_steps=max_episode_steps,
        )

        # import our training environment
        from openai_ros.task_envs.parrotdrone import parrotdrone_goto
    
    elif task_env == 'ParrotDroneTaskJ-v0':

        register(
            id=task_env,
            entry_point='openai_ros.task_envs.josutask.josutask:ParrotDroneTask',
            max_episode_steps=max_episode_steps,
        )

        # import our training environment
        from openai_ros.task_envs.josutask import josutask
    
    elif task_env == 'DroneXY-v3':

        register(
            id=task_env,
            entry_point='openai_ros.task_envs.drone_xy.drone_xy:DroneXY',
            max_episode_steps=max_episode_steps,
        )

        # import our training environment
        from openai_ros.task_envs.drone_xy import drone_xy
    
    elif task_env == 'DroneXYZ-v0':

        register(
            id=task_env,
            entry_point='openai_ros.task_envs.drone_xyz.drone_xyz:DroneXYZ',
            max_episode_steps=max_episode_steps,
        )

        # import our training environment
        from openai_ros.task_envs.drone_xyz import drone_xyz
   
    elif task_env == 'Zirkuitoa-v0':

        register(
            id=task_env,
            entry_point='openai_ros.task_envs.zirkuitoa.zirkuitoa:Zirkuitoa',
            max_episode_steps=max_episode_steps,
        )

        # import our training environment
        from openai_ros.task_envs.zirkuitoa import zirkuitoa

    
    elif task_env == 'MyTurtleBot2Maze-v0':

        register(
            id=task_env,
            entry_point='openai_ros.task_envs.turtlebot2.turtlebot2_maze:TurtleBot2MazeEnv',
            max_episode_steps=max_episode_steps,
        )
        # import our training environment
        from openai_ros.task_envs.turtlebot2 import turtlebot2_maze

    elif task_env == 'MyTurtleBot2Wall-v0':

        register(
            id=task_env,
            entry_point='openai_ros.task_envs.turtlebot2.turtlebot2_wall:TurtleBot2WallEnv',
            max_episode_steps=max_episode_steps,
        )
        # import our training environment
        from openai_ros.task_envs.turtlebot2 import turtlebot2_wall

    elif task_env == 'TurtleBot3World-v0':

        register(
            id=task_env,
            entry_point='openai_ros.task_envs.turtlebot3.turtlebot3_world:TurtleBot3WorldEnv',
            max_episode_steps=max_episode_steps,
        )

        # import our training environment
        from openai_ros.task_envs.turtlebot3 import turtlebot3_world


    # Add here your Task Envs to be registered
    else:
        result = False

    ###########################################################################

    if result:
        # We check that it was really registered
        supported_gym_envs = GetAllRegisteredGymEnvs()
        #print("REGISTERED GYM ENVS===>"+str(supported_gym_envs))
        assert (task_env in supported_gym_envs), "The Task_Robot_ENV given is not Registered ==>" + \
            str(task_env)

    return result


def GetAllRegisteredGymEnvs():
    """
    Returns a List of all the registered Envs in the system
    return EX: ['Copy-v0', 'RepeatCopy-v0', 'ReversedAddition-v0', ... ]
    """

    all_envs = envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]

    return env_ids
