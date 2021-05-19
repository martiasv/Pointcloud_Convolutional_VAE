#!/usr/bin/env python

import time
import rospy
from std_srvs.srv import Trigger, Empty, EmptyRequest
from pointcloud_utils.srv import scramble
import numpy as np
from gazebo_msgs.msg import ContactsState, ModelState




class EnvScrambler:
    def __init__(self):
        self.model_state_publisher = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        self.service = rospy.Service('world_randomize_service',scramble, self.randomize_world_obstacles)

        self.pause_physics_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_physics_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

        self.x_range = [-8,8]
        self.y_range = [-1.5,1.5]
        self.z_rot_range = [0,6.28]
        self.num_obstacles = 3
        self.num_type_obstacles = 6
        self.base_names = ['medium_boxes_','large_boxes_','medium_cylinders_','large_cylinders_']
        self.min_distance = 4

    def place_obstacle(self,obstacle_name,position,rotation):
        time.sleep(0.05) # need time between different publishings
        new_position = ModelState()
        new_position.model_name = obstacle_name
        new_position.reference_frame = 'world'
        new_position.pose.position.x = position[0]
        new_position.pose.position.y = position[1]
        new_position.pose.position.z = position[2]
        new_position.pose.orientation.x = 0
        new_position.pose.orientation.y = 0
        new_position.pose.orientation.z = rotation
        new_position.pose.orientation.w = 1
        self.model_state_publisher.publish(new_position)
        rospy.loginfo(f'Changing position of the obstacle {obstacle_name}')
        

    def reset_world_elements(self):
        counter = 0
        dx = 6
        for i in range(self.num_obstacles):
            self.place_obstacle(f'large_box_{i}',[(counter%3)*dx,8,1.25],0)
            self.place_obstacle(f'medium_box_{i}',[(counter%3)*dx,12,1],0)
            #self.place_obstacle(f'large_sphere_{i}',[(counter%3)*dx,16,1.5],0)
            #self.place_obstacle(f'medium_sphere_{i}',[(counter%3)*dx,20,1],0)
            self.place_obstacle(f'large_cylinder_{i}',[(counter%3)*dx,24,2.5],0)
            self.place_obstacle(f'medium_cylinder_{i}',[(counter%3)*dx,28,1.5],0)
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
                random_number = np.random.uniform([self.x_range[0],self.y_range[0],self.z_rot_range[0]],[self.x_range[1],self.y_range[1],self.z_rot_range[1]])
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
        random_number = np.random.uniform([self.x_range[0],self.y_range[0],self.z_rot_range[0]],[self.x_range[1],self.y_range[1],self.z_rot_range[1]])
        self.place_obstacle('large_box_0', [random_number[0],random_number[1],1.5], random_number[2])

    def medium_world_scrambler(self):
        random_positions = self.random_number_list_generator(4)
        self.place_obstacle('medium_box_0', [random_positions[0][0],random_positions[0][1],1], random_positions[0][2])
        #self.place_obstacle('medium_box_1', [random_positions[1][0],random_positions[1][1],1], random_positions[1][2])
        self.place_obstacle('medium_cylinder_0', [random_positions[2][0],random_positions[2][1],1.5], random_positions[2][2])
        self.place_obstacle('medium_cylinder_1', [random_positions[3][0],random_positions[3][1],1.5], random_positions[3][2])


    def hard_world_scrambler(self):
        random_positions = self.random_number_list_generator(4)
        self.place_obstacle('medium_box_0', [random_positions[0][0],random_positions[0][1],1], random_positions[0][2])
        self.place_obstacle('medium_box_1', [random_positions[1][0],random_positions[1][1],1], random_positions[1][2])
        self.place_obstacle('medium_cylinder_0', [random_positions[2][0],random_positions[2][1],1.5], random_positions[2][2])
        self.place_obstacle('large_cylinder_1', [random_positions[3][0],random_positions[3][1],2.5], random_positions[3][2])
        #self.place_obstacle('medium_sphere_0', [random_positions[4][0],random_positions[4][1],1], random_positions[4][2])
        #self.place_obstacle('medium_sphere_1', [random_positions[5][0],random_positions[5][1],1], random_positions[5][2])

    def randomize_world_obstacles(self,req):
        #Pause physics
        self.pause_physics_proxy(EmptyRequest())
        self.reset_world_elements()

        if req.env_to_scramble == 'easy':
            self.easy_world_scrambler()
        elif req.env_to_scramble == 'medium':
            self.medium_world_scrambler()
        elif req.env_to_scramble == 'hard':
            self.hard_world_scrambler()
        else:
            self.unpause_physics_proxy(EmptyRequest())
            return (0,'Please specify what type of world to scramble (easy/medium/hard)')

        #Unpause physics
        self.unpause_physics_proxy(EmptyRequest())
        return (1,f'Successfully scrambled {req.env_to_scramble} world!')


if __name__ == "__main__":
    rospy.init_node('world_randomizer')
    EnvScrambler()
    rospy.loginfo("Successfully made the service")
    rospy.spin()
    