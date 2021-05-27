#!/usr/bin/env python

import time
import rospy
from std_srvs.srv import Trigger, Empty, EmptyRequest
from pointcloud_utils.srv import scramble
import numpy as np
from scipy.spatial.transform import Rotation as R
from gazebo_msgs.msg import ContactsState, ModelState


class EnvScrambler:
    def __init__(self):
        self.model_state_publisher = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        self.service = rospy.Service('world_randomize_service',scramble, self.randomize_world_obstacles)
        self.reset_service = rospy.Service('world_reset_service',Trigger, self.reset_world_elements)


        self.pause_physics_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_physics_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

        self.x_range_easy = [-1,1]
        self.x_range_medium_hard = [-8,8]
        self.y_range = [-2,2]
        self.z_rot_range = [0,6.28]
        #self.z_rot_range = [0,0.01]
        self.num_obstacles = 3
        self.num_type_obstacles = 6
        self.base_names = ['medium_boxes_','large_boxes_','medium_cylinders_','large_cylinders_']
        self.min_distance = 4
        #x-value, y-value, z-value, rotation z#
        self.medium_box_0 = [[6.58244, 1.76233, 3, 2.44355],        [-7.01584, 0.866404, 3, 0.918785],  [3.70825, 0.907018, 3, 2.32801],     [3.09567, 1.83807, 3, 2.72694],    [-1.3884, 0.533599, 3, 2.69615],    [-3.92577, 1.81822, 3, 2.39968],    [-7.99825, 0.214869, 3, 0.145559]]
        self.medium_box_1 = [[-7.57629, -0.946075, 3, 2.68226],     [6.22798, 1.0296, 3, 1.93108],      [-1.63558, 1.04742, 3, 1.14769],     [-5.55521, -1.44665, 3, 2.69737],  [-6.42202, -0.471239, 3, 1.10473],  [0.179909, 0.77188, 3, 2.7307],     [1.47295, -1.97694, 3, 2.4298]]
        self.medium_cylinder_0 = [[-2.85481,-0.055442, 3.5, 2.37227],[-2.69751, -1.49152, 3.5, 2.80648], [-4.53812, -1.99743, 3.5, 1.04908], [0.125928, -1.28517, 3.5, 1.57871], [6.83406, -0.563981, 3.5, 2.7331], [7.63339, -1.59557, 3.5, 2.81921],  [-2.29815, 0.532679, 3.5, 2.1864]]
        self.large_cylinder_1 = [[2.37207, 1.60714, 4, 2.47031],    [1.01391, 0.224836, 4, 2.07925],    [7.1677, -1.86466, 4, 2.79313],      [7.47886, -1.94338, 4, 2.57798],   [2.83679, 0.453041, 4, 2.09261],    [-7.65464, -1.40533, 4, 1.21052],   [7.11053, 1.40976, 4, 2.6902]]

    def place_obstacle(self,obstacle_name,position,rotation):
        time.sleep(0.05) # need time between different publishings

        rotation_euler = R.from_euler('z',rotation, degrees=False)
        rotation_quat = rotation_euler.as_quat()

        new_position = ModelState()
        new_position.model_name = obstacle_name
        new_position.reference_frame = 'world'
        new_position.pose.position.x = position[0]
        new_position.pose.position.y = position[1]
        new_position.pose.position.z = position[2]
        new_position.pose.orientation.x = rotation_quat[0]
        new_position.pose.orientation.y = rotation_quat[1]
        new_position.pose.orientation.z = rotation_quat[2]
        new_position.pose.orientation.w = rotation_quat[3]
        self.model_state_publisher.publish(new_position)
        rospy.loginfo(f'Changing position of the obstacle {obstacle_name}')
        

    def reset_world_elements(self):
        counter = 0
        dx = 6
        for i in range(self.num_obstacles):
            self.place_obstacle(f'large_box_{i}',[(counter%3)*dx,8,2.5],0)
            self.place_obstacle(f'medium_box_{i}',[(counter%3)*dx,12,3],0)
            #self.place_obstacle(f'large_sphere_{i}',[(counter%3)*dx,16,1.5],0)
            #self.place_obstacle(f'medium_sphere_{i}',[(counter%3)*dx,20,1],0)
            self.place_obstacle(f'large_cylinder_{i}',[(counter%3)*dx,24,4],0)
            self.place_obstacle(f'medium_cylinder_{i}',[(counter%3)*dx,28,3.5],0)
            counter +=1

    def random_number_list_generator(self,num_elements):
        #Draw random numbers
        obst_num_counter = 0
        idx_counter = 0
        random_positions = []
        fail_ctr = 0
        while(len(random_positions) < num_elements):
            #Draw a random number for each obstacle and make sure that they have proper distance to each other.
            invalid_number = True
            #print("Defining valid_number = False")
            while(invalid_number):
                if fail_ctr == 30:
                    #"Start a fresh list if no candidate can be found"
                    fail_ctr = 0
                    random_positions = []
                #print("Drawing a random number")
                random_number = np.random.uniform([self.x_range_medium_hard[0],self.y_range[0],self.z_rot_range[0]],[self.x_range_medium_hard[1],self.y_range[1],self.z_rot_range[1]])
                if len(random_positions)==0:
                    random_positions.append(random_number)
                for j in range(len(random_positions)):
                    distance = np.sqrt((random_positions[j][0]-random_number[0])**2+ (random_positions[j][1]-random_number[1])**2)
                    #print(f'Distance between current random number and number {j} is: {distance}')
                    #print(f'...based on the following equation: np.sqrt(({random_positions[j][0]}-{random_number[0]})**2+({random_positions[j][1]}-{random_number[1]})**2)')
                    
                    #If the number generator has passed through the whole list and the number matches, we increment a counter. This is to prevent infinite loops where there exist no candidate for new points
                    if (distance <self.min_distance) and (j==len(random_positions)-1):
                        fail_ctr = fail_ctr + 1

                    if (distance < self.min_distance):
                        #print("Number is too close, drawing new number")
                        break
                    elif (j==len(random_positions)-1):
                        #print("No other obstacles is within the given radius. Adding number to list")
                        random_positions.append(random_number)
                        invalid_number = False
        return random_positions

    def easy_world_scrambler(self):
        random_number = np.random.uniform([self.x_range_easy[0],self.y_range[0],self.z_rot_range[0]],[self.x_range_easy[1],self.y_range[1],self.z_rot_range[1]])
        self.place_obstacle('large_box_0', [random_number[0],random_number[1],1.5], random_number[2])

    def medium_world_scrambler(self):
        random_positions = self.random_number_list_generator(4)
        self.place_obstacle('medium_box_0', [random_positions[0][0],random_positions[0][1],2], random_positions[0][2])
        #self.place_obstacle('medium_box_1', [random_positions[1][0],random_positions[1][1],1], random_positions[1][2])
        self.place_obstacle('medium_cylinder_0', [random_positions[2][0],random_positions[2][1],1.5], random_positions[2][2])
        self.place_obstacle('medium_cylinder_1', [random_positions[3][0],random_positions[3][1],1.5], random_positions[3][2])


    def hard_world_scrambler(self):
        random_positions = self.random_number_list_generator(4)
        self.place_obstacle('medium_box_0', [random_positions[0][0],random_positions[0][1],3], random_positions[0][2])
        self.place_obstacle('medium_box_1', [random_positions[1][0],random_positions[1][1],3], random_positions[1][2])
        self.place_obstacle('medium_cylinder_0', [random_positions[2][0],random_positions[2][1],3.5], random_positions[2][2])
        self.place_obstacle('large_cylinder_1', [random_positions[3][0],random_positions[3][1],4],   random_positions[3][2])
        #self.place_obstacle('medium_sphere_0', [random_positions[4][0],random_positions[4][1],1], random_positions[4][2])
        #self.place_obstacle('medium_sphere_1', [random_positions[5][0],random_positions[5][1],1], random_positions[5][2])

    def set_world_scrambler(self,number):
        world_number = number -1 
        self.place_obstacle('medium_box_0', [self.medium_box_0[world_number][0],self.medium_box_0[world_number][1],self.medium_box_0[world_number][2]], self.medium_box_0[world_number][3])
        self.place_obstacle('medium_box_1', [self.medium_box_1[world_number][0],self.medium_box_1[world_number][1],self.medium_box_1[world_number][2]], self.medium_box_1[world_number][3])
        self.place_obstacle('medium_cylinder_0', [self.medium_cylinder_0[world_number][0],self.medium_cylinder_0[world_number][1],self.medium_cylinder_0[world_number][2]], self.medium_cylinder_0[world_number][3])
        self.place_obstacle('large_cylinder_1', [self.large_cylinder_1[world_number][0],self.large_cylinder_1[world_number][1],self.large_cylinder_1[world_number][2]], self.large_cylinder_1[world_number][3])

    def randomize_world_obstacles(self,req):
        #Pause physics
        self.reset_world_elements()
        print("\n\n\n\n This is the message to the world scrambler: ",req.env_to_scramble[0:3])
        if req.env_to_scramble == 'easy':
            self.easy_world_scrambler()
        elif req.env_to_scramble == 'medium':
            self.medium_world_scrambler()
        elif req.env_to_scramble == 'hard':
            self.hard_world_scrambler()
        elif req.env_to_scramble[0:3] == 'set':
            self.set_world_scrambler(int(req.env_to_scramble[-1]))
        else:
            self.unpause_physics_proxy(EmptyRequest())
            return (0,'Please specify what type of world to scramble (easy/medium/hard)')

        #Unpause physics
        
        return (1,f'Successfully scrambled {req.env_to_scramble} world!')


if __name__ == "__main__":
    rospy.init_node('world_randomizer')
    EnvScrambler()
    rospy.loginfo("Successfully made the service")
    rospy.spin()
    