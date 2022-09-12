#!/usr/bin/env python3

import gym
import numpy
import time
import qlearn
from gym import wrappers
import functools
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

    # Ikasketa Parametroak kargatu 
    # Parametroak config karpetako .yaml fitxategian gordeta daude
    # Launch fitxategia jaurtitzean kargatzen dira
    Alpha = rospy.get_param("/drone/alpha")
    Epsilon = rospy.get_param("/drone/epsilon")
    Gamma = rospy.get_param("/drone/gamma")
    epsilon_discount = rospy.get_param("/drone/epsilon_discount")
    nepisodes = rospy.get_param("/drone/nepisodes")
    nsteps = rospy.get_param("/drone/nsteps")

    # Ikasteko erabiliko den algoritmoa hasieratu
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    start_time = time.time()
    highest_reward = 0
    
    #Begiratu ea aurretiko ikasketarik badagoen eta kargatu baldin badago:
    qfile = 'taula.npy'
    if(os.path.exists(qfile)):  #os.path.exists(qfile)
    	print(qfile," fitxategitik kargatzen...")
    	qlearn.load(qfile)
    else:
    	print("------------- Ez da fitxategirik aurkitu --------------")
    	
    # Entrenamendu begizta nagusia: x aldagaiarekin episodioen jarraipena 
    for x in range(nepisodes):
        rospy.logdebug("############### EPISODIOA HASI=>" + str(x))

        cumulated_reward = 0
        done = False
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        # Ingurunea hasieratu eta dronearen lehenengo egoera lortu
        observation = env.reset()
        state = ''.join(map(str, observation))

        # Show on screen the actual situation of the robot
        # env.render()
        # Episodio bakoitzean nsteps pausu egin dronearekin
        for i in range(nsteps):
            rospy.logwarn("############### PAUSOA HASI=>" + str(i))
            # Momentuko egoeraren arabera ekintza bat aukeratu
            action = qlearn.chooseAction(state)
            rospy.logwarn("Hurrengo ekintza:%d", action)
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)

            rospy.logwarn(str(observation) + " " + str(reward))
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            # Make the algorithm learn based on the results
            rospy.logwarn("# aurreko egoera=>      " + str(state))
            rospy.logwarn("# harturiko ekintza=>   " + str(action))
            rospy.logwarn("# emandako saria=>     " + str(reward))
            rospy.logwarn("# Sari totala => 	   " +
                          str(cumulated_reward))
            rospy.logwarn(
                "# Hurrengo pausoa hasteko egoera=>" + str(nextState))
            qlearn.learn(state, action, reward, nextState)
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
        qlearn.save(qfile)
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logerr(("EP: " + str(x + 1) + " - [alpha: " + str(round(qlearn.alpha, 2)) + " - gamma: " + str(
            round(qlearn.gamma, 2)) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Saria: " + str(
            cumulated_reward) + "     Denbora: %d:%02d:%02d" % (h, m, s)))
        
    rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
        initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(
        functools.reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))
    qlearn.save(qfile)
    env.close()
