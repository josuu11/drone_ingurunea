#!/usr/bin/env python3
import functools
import gym
import numpy
import time
import sarsa  # qlearn ordez sarsa ezarri da
from gym import wrappers
# ROS packages required
import rospy
import rospkg
import os
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment


if __name__ == '__main__':

    rospy.init_node('parrotdrone_goto_qlearn',
                    anonymous=True, log_level=rospy.WARN)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/drone/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    rospy.loginfo("Gym environment done")

    # Ikasketa jarraipena egiteko sistema ezarri
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('josudrone') # Emaitzak pakete honen barruan gorde
    outdir = pkg_path + '/emaitzak' # Emaitzak gordeko diren helbidea
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # Ikasketa parametroak kargatu
    # Parametroak config karpetako .yaml fitxategian gordeta daude
    # Launch fitxategia jaurtitzean kargatzen dira
    Alpha = rospy.get_param("/drone/alpha")
    Epsilon = rospy.get_param("/drone/epsilon")
    Gamma = rospy.get_param("/drone/gamma")
    epsilon_discount = rospy.get_param("/drone/epsilon_discount")
    nepisodes = rospy.get_param("/drone/nepisodes")
    nsteps = rospy.get_param("/drone/nsteps")

    # Ikasteko erabiliko den algoritmoa hasieratu
    sarsa = sarsa.Sarsa(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = sarsa.epsilon

    start_time = time.time()
    highest_reward = 0
    
    # Begiratu ea aurretiko ikasketarik badagoen eta kargatu baldin badago:
    qfile = 'taula.npy'
    if(os.path.exists(qfile)):  #os.path.exists(qfile)
    	print(qfile," fitxategitik kargatzen...")
    	sarsa.load(qfile)
    else:
    	print("------------- Ez da fitxategirik aurkitu --------------")
    	
    # Entrenamendu begizta nagusia: x aldagaiarekin episodioen jarraipena 
    for x in range(nepisodes):
        rospy.logdebug("############### EPISODIOA HASI=>" + str(x))

        cumulated_reward = 0
        done = False
        if sarsa.epsilon > 0.05:
            sarsa.epsilon *= epsilon_discount

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = ''.join(map(str, observation))

        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot for nsteps
        for i in range(nsteps):
            rospy.logwarn("############### START STEP=>" + str(i))
            # Pick an action based on the current state
            action = sarsa.chooseAction(state)
            rospy.logwarn("Hurrengo ekintza:%d", action)
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)

            rospy.logwarn(str(observation) + " " + str(reward))
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))
            nextAction = sarsa.chooseAction(nextState)
            # Make the algorithm learn based on the results
            rospy.logwarn("# aurreko egoera=>" + str(state))
            rospy.logwarn("# harturiko ekintza=>" + str(action))
            rospy.logwarn("# ekintzak emandako saria=>" + str(reward))
            rospy.logwarn("# Sari totala=>" +
                          str(cumulated_reward))
            rospy.logwarn(
                "# Hurrengo pausoa hasteko egoera=>" + str(nextState))
            sarsa.learn(state, action, reward, nextState, nextAction)

            if not (done):
                rospy.logwarn("NOT DONE")
                state = nextState
            elif i == (nsteps-1):
                rospy.logwarn("---------Pauso max--------")
                break
            else:
                rospy.logwarn("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
            rospy.logwarn("############### END Step=>" + str(i))
            #raw_input("Next Step...PRESS KEY")
            # rospy.sleep(2.0)
        sarsa.save(qfile)      
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logerr(("EP: " + str(x + 1) + " - [alpha: " + str(round(sarsa.alpha, 2)) + " - gamma: " + str(
            round(sarsa.gamma, 2)) + " - epsilon: " + str(round(sarsa.epsilon, 2)) + "] - Saria: " + str(
            cumulated_reward) + "     Denbora: %d:%02d:%02d" % (h, m, s)))

    rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(sarsa.alpha) + "|" + str(sarsa.gamma) + "|" + str(
        initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(
        functools.reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
