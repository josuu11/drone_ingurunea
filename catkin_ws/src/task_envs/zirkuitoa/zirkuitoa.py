import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import parrotdrone_env
from gym.envs.registration import register
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
from tf.transformations import euler_from_quaternion
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
import os

class Zirkuitoa(parrotdrone_env.ParrotDroneEnv):
    def __init__(self):
        """
        Parrot dronea puntu batetik bestera joateko entrenatu
        """
        ros_ws_abspath = rospy.get_param("/drone/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        ROSLauncher(rospackage_name="drone_construct",	# Mundua gordetzen den paketearen izena
                    launch_file_name="start_world.launch",	# Mundua jaurtitzeko fitxategia
                    ros_ws_abspath=ros_ws_abspath)

        # Parametroak kargatu .yaml fitxategitik
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/zirkuitoa/config", # Konfigurazio fitxategiaren helbidea
                               yaml_file_name="config.yaml")	# Konfigurazio fitxategia

        # Only variable needed to be set here
        number_actions = rospy.get_param('/drone/n_actions')
        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)

        # Actions and Observations
        self.linear_forward_speed = rospy.get_param(
            '/drone/linear_forward_speed')
        self.angular_turn_speed = rospy.get_param('/drone/angular_turn_speed')
        self.angular_speed = rospy.get_param('/drone/angular_speed')

        self.init_linear_speed_vector = Vector3()
        self.init_linear_speed_vector.x = rospy.get_param(
            '/drone/init_linear_speed_vector/x')
        self.init_linear_speed_vector.y = rospy.get_param(
            '/drone/init_linear_speed_vector/y')
        self.init_linear_speed_vector.z = rospy.get_param(
            '/drone/init_linear_speed_vector/z')

        self.init_angular_turn_speed = rospy.get_param(
            '/drone/init_angular_turn_speed')

        self.max_linear_aceleration = rospy.get_param('/drone/max_linear_aceleration')

        self.min_sonar_value = rospy.get_param('/drone/min_sonar_value')
        self.max_sonar_value = rospy.get_param('/drone/max_sonar_value')

        # Ibilbideko mugak kargatu
        self.work_space_x_max1 = rospy.get_param("/drone/ibilbide_mugak/x_max1")
        self.work_space_x_min1 = rospy.get_param("/drone/ibilbide_mugak/x_min1")
        self.work_space_y_max1 = rospy.get_param("/drone/ibilbide_mugak/y_max1")
        self.work_space_y_min1 = rospy.get_param("/drone/ibilbide_mugak/y_min1")
        self.work_space_z_max1 = rospy.get_param("/drone/ibilbide_mugak/z_max1")
        self.work_space_z_min1 = rospy.get_param("/drone/ibilbide_mugak/z_min1")
        self.work_space_x_max2 = rospy.get_param("/drone/ibilbide_mugak/x_max2")
        self.work_space_x_min2 = rospy.get_param("/drone/ibilbide_mugak/x_min2")
        self.work_space_y_max2 = rospy.get_param("/drone/ibilbide_mugak/y_max2")
        self.work_space_y_min2 = rospy.get_param("/drone/ibilbide_mugak/y_min2")
        self.work_space_z_max2 = rospy.get_param("/drone/ibilbide_mugak/z_max2")
        self.work_space_z_min2 = rospy.get_param("/drone/ibilbide_mugak/z_min2")
        self.work_space_x_max3 = rospy.get_param("/drone/ibilbide_mugak/x_max3")
        self.work_space_x_min3 = rospy.get_param("/drone/ibilbide_mugak/x_min3")
        self.work_space_y_max3 = rospy.get_param("/drone/ibilbide_mugak/y_max3")
        self.work_space_y_min3 = rospy.get_param("/drone/ibilbide_mugak/y_min3")
        self.work_space_z_max3 = rospy.get_param("/drone/ibilbide_mugak/z_max3")
        self.work_space_z_min3 = rospy.get_param("/drone/ibilbide_mugak/z_min3")

        # Maximum RPY values
        self.max_roll = rospy.get_param("/drone/max_roll")
        self.max_pitch = rospy.get_param("/drone/max_pitch")
        self.max_yaw = rospy.get_param("/drone/max_yaw")

        # Zirkuitoko puntuak
        # P1
        self.desired_point1 = Point()
        self.desired_point1.x = rospy.get_param("/drone/p1/x")
        self.desired_point1.y = rospy.get_param("/drone/p1/y")
        self.desired_point1.z = rospy.get_param("/drone/p1/z")
        # P2
        self.desired_point2 = Point()
        self.desired_point2.x = rospy.get_param("/drone/p2/x")
        self.desired_point2.y = rospy.get_param("/drone/p2/y")
        self.desired_point2.z = rospy.get_param("/drone/p2/z")
        # Helmuga
        self.desired_point = Point()
        self.desired_point.x = rospy.get_param("/drone/helmuga/x")
        self.desired_point.y = rospy.get_param("/drone/helmuga/y")
        self.desired_point.z = rospy.get_param("/drone/helmuga/z")

        self.desired_point_epsilon = rospy.get_param(
            "/drone/desired_point_epsilon")

        # We place the Maximum and minimum values of the X,Y,Z,R,P,Yof the pose

        high = numpy.array([self.work_space_x_max1,
                            self.work_space_y_max1,
                            self.work_space_z_max1,
                            self.max_roll,
                            self.max_pitch,
                            self.max_yaw,
                            self.max_sonar_value])

        low = numpy.array([self.work_space_x_min1,
                           self.work_space_y_min1,
                           self.work_space_z_min1,
                           -1*self.max_roll,
                           -1*self.max_pitch,
                           -numpy.inf,
                           self.min_sonar_value])

        self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>" +
                       str(self.observation_space))

        # Sariak
        self.pauso_saria = rospy.get_param("/drone/pauso_eman")
        self.gerturatu = rospy.get_param("/drone/gerturatu")
        self.p1_iritsi = rospy.get_param("/drone/p1_iritsi")
        self.p2_iritsi = rospy.get_param("/drone/p2_iritsi")
        self.helm_iritsi = rospy.get_param("/drone/helmugara_iritsi")
        self.mugatik_atera = rospy.get_param("/drone/ibilbidetik_atera")

        self.cumulated_steps = 0.0
        self.p1_iritsi_da = False	# P1 puntura iritsi den edo ez jakiteko
        self.p2_iritsi_da = False	# P2 puntura iritsi den edo ez jakiteko
        # Here we will add any init functions prior to starting the MyRobotEnv
        super(Zirkuitoa, self).__init__(ros_ws_abspath)

    def _set_init_pose(self):
        """
        Sets the Robot in its init linear and angular speeds
        and lands the robot. Its preparing it to be reseted in the world.
        """
        #raw_input("INIT SPEED PRESS")
        self.move_base(self.init_linear_speed_vector,
                       self.init_angular_turn_speed,
                       epsilon=0.05,
                       update_rate=10)
        # We Issue the landing command to be sure it starts landing
        #raw_input("LAND PRESS")
        # self.land()
        self.p1_iritsi_da = False
        self.p2_iritsi_da = False

        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        #raw_input("TakeOFF PRESS")
        # We TakeOff before sending any movement commands
        self.takeoff()

        # For Info Purposes
        self.cumulated_reward = 0.0

        # We get the initial pose to mesure the distance from the desired point.
        gt_pose = self.get_gt_pose()
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(
            gt_pose.position)

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the drone
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """

        rospy.logdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the parent class of Parrot
        linear_speed_vector = Vector3()
        angular_speed = 0.0

        if action == 0:  # FORWARDS
            linear_speed_vector.x = self.linear_forward_speed
            self.last_action = "FORWARDS"
        elif action == 1:  # BACKWARDS
            linear_speed_vector.x = -1*self.linear_forward_speed
            self.last_action = "BACKWARDS"
        elif action == 2:  # STRAFE_LEFT
            linear_speed_vector.y = self.linear_forward_speed
            self.last_action = "STRAFE_LEFT"
        elif action == 3:  # STRAFE_RIGHT
            linear_speed_vector.y = -1*self.linear_forward_speed
            self.last_action = "STRAFE_RIGHT"
        """
        elif action == 4:  # UP
            linear_speed_vector.z = self.linear_forward_speed
            self.last_action = "UP"
        
        elif action == 5:  # DOWN
            linear_speed_vector.z = -1*self.linear_forward_speed
            self.last_action = "DOWN"
        """
        # We tell drone the linear and angular speed to set to execute
        self.move_base(linear_speed_vector,
                       angular_speed,
                       epsilon=0.05,
                       update_rate=10)

        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        droneEnv API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        gt_pose = self.get_gt_pose()

        # We get the orientation of the cube in RPY
        roll, pitch, yaw = self.get_orientation_euler(gt_pose.orientation)

        # We get the sonar value
        sonar = self.get_sonar()
        sonar_value = sonar.range

        """
        observations = [    round(gt_pose.position.x, 1),
                            round(gt_pose.position.y, 1),
                            round(gt_pose.position.z, 1),
                            round(roll, 1),
                            round(pitch, 1),
                            round(yaw, 1),
                            round(sonar_value,1)]
        """
        # We simplify a bit the spatial grid to make learning faster
        observations = [int(gt_pose.position.x),
                        int(gt_pose.position.y),
                        int(gt_pose.position.z),
                        round(roll, 1),
                        round(pitch, 1),
                        round(yaw, 1),
                        round(sonar_value, 1)]

        rospy.logdebug("Observations==>"+str(observations))
        rospy.logdebug("END Get Observation ==>")
        return observations

    def _is_done(self, observations):
        """
        Episodioa 4 baldintzengatik bukatu:
        1) Lan eremutik kanpo dago
        2) Sonar-ak zerbait gertuegi detektatu
        3) Biratu egin da talka baten gatik
        4) Helmugara iritsi da
        """

        episode_done = False

        current_position = Point()
        current_position.x = observations[0]
        current_position.y = observations[1]
        current_position.z = observations[2]

        current_orientation = Point()
        current_orientation.x = observations[3]
        current_orientation.y = observations[4]
        current_orientation.z = observations[5]

        sonar_value = observations[6]
        is_inside_workspace_now = self.is_inside_workspace(current_position)
        sonar_detected_something_too_close_now = self.sonar_detected_something_too_close(
            sonar_value)
        drone_flipped = self.drone_has_flipped(current_orientation)
        has_reached_des_point = self.is_in_desired_position(
            current_position, self.desired_point_epsilon, 0)
        
        rospy.logwarn(">>>>>> EMAITZAK <<<<<")
        if not is_inside_workspace_now:
            rospy.logerr("Lan eremuan dago = " +
                         str(is_inside_workspace_now))
        else:
            rospy.logwarn("Lan eremuan dago = " +
                          str(is_inside_workspace_now))

        if sonar_detected_something_too_close_now:
            rospy.logerr("Sonar-ak zerbait gertu detektatu = " +
                         str(sonar_detected_something_too_close_now))
        else:
            rospy.logwarn("Sonar-ak zerbait gertu detektatu = " +
                          str(sonar_detected_something_too_close_now))

        if drone_flipped:
            rospy.logerr("Dronea biratuta = "+str(drone_flipped))
        else:
            rospy.logwarn("Dronea biratuta = "+str(drone_flipped))

        if has_reached_des_point:
            rospy.logerr("Helmugara iritsi da = "+str(has_reached_des_point))
        else:
            rospy.logwarn("Helmugara iritsi da = "+str(has_reached_des_point))
        
        # We see if we are outside the Learning Space
        episode_done = not(
            is_inside_workspace_now) or sonar_detected_something_too_close_now or drone_flipped or has_reached_des_point

        if episode_done:
            rospy.logerr("epsiodioa amaitu da====>"+str(episode_done))
        else:
            rospy.logwarn("epsiodioa amaitu da====>"+str(episode_done))

        return episode_done

    def _compute_reward(self, observations, done):

        current_position = Point()
        current_position.x = observations[0]
        current_position.y = observations[1]
        current_position.z = observations[2]

        distance_from_des_point = self.get_distance_from_desired_point(current_position)
            
        distance_difference = distance_from_des_point - self.previous_distance_from_des_point
            

        if not done:

            if self.is_in_desired_position(current_position, self.desired_point_epsilon,1) and not self.p1_iritsi_da:
                reward = self.p1_iritsi
            elif self.is_in_desired_position(current_position, self.desired_point_epsilon,2) and not self.p2_iritsi_da:
                reward = self.p2_iritsi
            else:
                if distance_difference < 0.0:
                    rospy.logwarn("DISTANTZIA HANDITU DA")
                    reward = self.gerturatu
                else:
                    rospy.logerr("DISTANTZIA TXIKITU DA")
                    reward = self.pauso_saria
        else:

            if self.is_in_desired_position(current_position, self.desired_point_epsilon):
                reward = self.helm_iritsi
            else:
                reward = self.mugatik_atera

        return reward

    # Internal TaskEnv Methods

    def is_in_desired_position(self, current_position, epsilon=0.05, puntua=0):
        """
        True bueltatu dronea helmugako posizioan badago
        """

        is_in_desired_pos = False
        if puntua == 1:
                x_pos_plus = self.desired_point1.x + epsilon
                x_pos_minus = self.desired_point1.x - epsilon
                y_pos_plus = self.desired_point1.y + epsilon
                y_pos_minus = self.desired_point1.y - epsilon
        elif puntua == 2:
                x_pos_plus = self.desired_point2.x + epsilon
                x_pos_minus = self.desired_point2.x - epsilon
                y_pos_plus = self.desired_point2.y + epsilon
                y_pos_minus = self.desired_point2.y - epsilon
        else:
                x_pos_plus = self.desired_point.x + epsilon
                x_pos_minus = self.desired_point.x - epsilon
                y_pos_plus = self.desired_point.y + epsilon
                y_pos_minus = self.desired_point.y - epsilon
        

        x_current = current_position.x
        y_current = current_position.y

        x_pos_are_close = (x_current <= x_pos_plus) and (
            x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (
            y_current > y_pos_minus)

        is_in_desired_pos = x_pos_are_close and y_pos_are_close

        rospy.logwarn("###### HELMUGAN DAGO ? ######")
        rospy.logwarn("Momentuko pos = \n"+str(current_position))
        rospy.logwarn("x_pos_plus = "+str(x_pos_plus) + " ,x_pos_minus = "+str(x_pos_minus))              
        rospy.logwarn("y_pos_plus = "+str(y_pos_plus) + " ,y_pos_minus = "+str(y_pos_minus))            
        rospy.logwarn("x_pos_are_close =  "+str(x_pos_are_close))
        rospy.logwarn("y_pos_are_close =  "+str(y_pos_are_close))
        rospy.logwarn("Helmugan dago = "+str(is_in_desired_pos))
        rospy.logwarn("############")

        return is_in_desired_pos

    def is_inside_workspace(self, current_position):
        """
        Dronea definitutako lan eremuan dagoen konprobatu
        """
        is_inside = False

        rospy.logwarn("##### LAN EREMU BARRUAN? #######")
        rospy.logwarn("XYZ momentuko pos \n"+str(current_position))

        if self.p1_iritsi_da:
            if self.p2_iritsi_da: 
               rospy.logwarn("lan_eremua_x_max="+str(self.work_space_x_max3) + ",lan_eremua_x_min="+str(self.work_space_x_min3))                 
               rospy.logwarn("lan_eremua_y_max="+str(self.work_space_y_max3) + ",lan_eremua_y_min="+str(self.work_space_y_min3))           
               rospy.logwarn("lan_eremua_z_max="+str(self.work_space_z_max3) + ",lan_eremua_z_min="+str(self.work_space_z_min3))            
               rospy.logwarn("############")
            else:
               rospy.logwarn("lan_eremua_x_max="+str(self.work_space_x_max2) + ",lan_eremua_x_min="+str(self.work_space_x_min2))                 
               rospy.logwarn("lan_eremua_y_max="+str(self.work_space_y_max2) + ",lan_eremua_y_min="+str(self.work_space_y_min2))           
               rospy.logwarn("lan_eremua_z_max="+str(self.work_space_z_max2) + ",lan_eremua_z_min="+str(self.work_space_z_min2))            
               rospy.logwarn("############")     
        else:
            rospy.logwarn("lan_eremua_x_max="+str(self.work_space_x_max1) + ",lan_eremua_x_min="+str(self.work_space_x_min1))                 
            rospy.logwarn("lan_eremua_y_max="+str(self.work_space_y_max1) + ",lan_eremua_y_min="+str(self.work_space_y_min1))           
            rospy.logwarn("lan_eremua_z_max="+str(self.work_space_z_max1) + ",lan_eremua_z_min="+str(self.work_space_z_min1))            
            rospy.logwarn("############")
        
        if self.p1_iritsi_da:
            if self.p2_iritsi_da:    
                if current_position.x > self.work_space_x_min3 and current_position.x <= self.work_space_x_max3:
                    if current_position.y > self.work_space_y_min3 and current_position.y <= self.work_space_y_max3:
                        if current_position.z > self.work_space_z_min3 and current_position.z <= self.work_space_z_max3:
                            is_inside = True
                            rospy.logwarn("###### p1 eta p2 true######")
            else:
                if current_position.x > self.work_space_x_min2 and current_position.x <= self.work_space_x_max2:
                    if current_position.y > self.work_space_y_min2 and current_position.y <= self.work_space_y_max2:
                        if current_position.z > self.work_space_z_min2 and current_position.z <= self.work_space_z_max2:
                            is_inside = True
                            rospy.logwarn("###### p1 true######")
        else:
            if current_position.x > self.work_space_x_min1 and current_position.x <= self.work_space_x_max1:
                if current_position.y > self.work_space_y_min1 and current_position.y <= self.work_space_y_max1:
                    if current_position.z > self.work_space_z_min1 and current_position.z <= self.work_space_z_max1:
                        is_inside = True
                        rospy.logwarn("###### hasierako mugatan######")

        return is_inside

    def sonar_detected_something_too_close(self, sonar_value):
        """
        Dronaren parean zerbait gertu dagoen detektatu
        """
        rospy.logwarn("##### SONAR ZERBAIT GERTU? #######")
        rospy.logwarn("sonar_balioa"+str(sonar_value) +
                      ",min_sonar_balioa="+str(self.min_sonar_value))
        rospy.logwarn("############")

        too_close = sonar_value < self.min_sonar_value

        return too_close

    def drone_has_flipped(self, current_orientation):
        """
        Based on the orientation RPY given states if the drone has flipped
        """
        has_flipped = True

        self.max_roll = rospy.get_param("/drone/max_roll")
        self.max_pitch = rospy.get_param("/drone/max_pitch")

        rospy.logwarn("#### BIRATU EGIN DA? ########")
        rospy.logwarn("RPY orientazioa"+str(current_orientation))
        rospy.logwarn("max_roll"+str(self.max_roll) +
                      ",min_roll="+str(-1*self.max_roll))
        rospy.logwarn("max_pitch"+str(self.max_pitch) +
                      ",min_pitch="+str(-1*self.max_pitch))
        rospy.logwarn("############")

        if current_orientation.x > -1*self.max_roll and current_orientation.x <= self.max_roll:
            if current_orientation.y > -1*self.max_pitch and current_orientation.y <= self.max_pitch:
                has_flipped = False

        return has_flipped

    def get_distance_from_desired_point(self, current_position):
        """
        Momentuko posiziotik helmugarako distantzia kalkulatu
        :param start_point:
        :return:
        """
        self.helburua = Point()
        """
        self.helburua.x = self.desired_point.x
        self.helburua.y = self.desired_point.y
        self.helburua.z = self.desired_point.z
        """
        if self.p1_iritsi_da:
            if self.p2_iritsi_da:    
                distance = self.get_distance_from_point(current_position,self.desired_point)
                self.helburua=self.desired_point
            else:
                distance = self.get_distance_from_point(current_position,self.desired_point2)
                self.helburua=self.desired_point2
            
        else:
            distance = self.get_distance_from_point(current_position,self.desired_point1)
            self.helburua=self.desired_point1
             
        rospy.logwarn("Momentuko Helburua: "+str(self.helburua.x)+" "+str(self.helburua.y)+" "+str(self.helburua.z))
        return distance

    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = numpy.array((pstart.x, pstart.y, pstart.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))

        distance = numpy.linalg.norm(a - b)

        return distance

    def get_orientation_euler(self, quaternion_vector):
        # We convert from quaternions to euler
        orientation_list = [quaternion_vector.x,
                            quaternion_vector.y,
                            quaternion_vector.z,
                            quaternion_vector.w]

        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw
        
    def get_vector_magnitude(self, vector):
        """
        It calculated the magnitude of the Vector3 given.
        This is usefull for reading imu accelerations and knowing if there has been
        a crash
        :return:
        """
        contact_force_np = numpy.array((vector.x, vector.y, vector.z))
        force_magnitude = numpy.linalg.norm(contact_force_np)

        return force_magnitude

