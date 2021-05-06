#!/usr/bin/env python

import time
import rospy
from std_srvs.srv import Trigger, Empty, EmptyRequest
import numpy as np
from gazebo_msgs.msg import ContactsState, ModelState

x_range = [-50,50]
y_range = [-4,4]
z_rot_range = [0,6.28]
num_obstacles = 8
num_type_obstacles = 9
base_names = ['small_spheres_','medium_spheres_','large_spheres_','small_boxes_','medium_boxes_','large_boxes_','small_cylinders_','medium_cylinders_','large_cylinders_']

class EnvScrambler:
    def __init__(self):
        self.model_state_publisher = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        self.service = rospy.Service('world_randomize_service', Trigger, self.randomize_world_obstacles)

        self.pause_physics_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_physics_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

    def randomly_place_obstacle(self,obstacle_name,position,rotation):
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
        time.sleep(0.03) # need time between different publishments  

    def randomize_world_obstacles(self,req):
        #Pause physics
        self.pause_physics_proxy(EmptyRequest())

        #Draw random numbers
        obst_num_counter = 0
        idx_counter = 0
        random_positions = []
        fail_ctr = 0
        while(len(random_positions) < num_obstacles*num_type_obstacles):
            #Draw a random number for each obstacle and make sure that they have proper distance to each other.
            invalid_number = True
            #print("Defining valid_number = False")
            while(invalid_number):
                if fail_ctr == 30:
                    #"Start a fresh list if no candidate can be found"
                    fail_ctr = 0
                    random_positions = []
                #print("Drawing a random number")
                random_number = np.random.uniform([x_range[0],y_range[0],z_rot_range[0]],[x_range[1],y_range[1],z_rot_range[1]])
                if len(random_positions)==0:
                    random_positions.append(random_number)
                for j in range(len(random_positions)):
                    distance = np.sqrt((random_positions[j][0]-random_number[0])**2+ (random_positions[j][1]-random_number[1])**2)
                    #print(f'Distance between current random number and number {j} is: {distance}')
                    #print(f'...based on the following equation: np.sqrt(({random_positions[j][0]}-{random_number[0]})**2+({random_positions[j][1]}-{random_number[1]})**2)')
                    
                    #If the number generator has passed through the whole list and the number matches, we increment a counter. This is to prevent infinite loops where there exist no candidate for new points
                    if (distance <3.0) and (j==len(random_positions)-1):
                        fail_ctr = fail_ctr + 1

                    if (distance < 3.0):
                        #print("Number is too close, drawing new number")
                        break
                    elif (j==len(random_positions)-1):
                        #print("No other obstacles is within the given radius. Adding number to list")
                        random_positions.append(random_number)
                        invalid_number = False

                    
    
        #random_positions = np.array([np.random.uniform([x_range[0],y_range[0],z_rot_range[0]],[x_range[1],y_range[1],z_rot_range[1]]) for x in range(num_obstacles*num_type_obstacles)])
        for i in range(num_obstacles):
            #Iterate through the small spheres
            self.randomly_place_obstacle(f'small_sphere_{obst_num_counter}', [random_positions[idx_counter][0],random_positions[idx_counter][1],0.5], random_positions[idx_counter][2])
            idx_counter +=1
            #Iterate through the medium spheres
            self.randomly_place_obstacle(f'medium_sphere_{obst_num_counter}', [random_positions[idx_counter][0],random_positions[idx_counter][1],1.0], random_positions[idx_counter][2])
            idx_counter += 1
            #Iterate through the large spheres
            self.randomly_place_obstacle(f'large_sphere_{obst_num_counter}', [random_positions[idx_counter][0],random_positions[idx_counter][1],1.5], random_positions[idx_counter][2])
            idx_counter += 1
            #Iterate through the small boxes
            self.randomly_place_obstacle(f'small_box_{obst_num_counter}', [random_positions[idx_counter][0],random_positions[idx_counter][1],0], random_positions[idx_counter][2])
            idx_counter += 1
            #Iterate through the medium boxes
            self.randomly_place_obstacle(f'medium_box_{obst_num_counter}', [random_positions[idx_counter][0],random_positions[idx_counter][1],0], random_positions[idx_counter][2])
            idx_counter += 1
            #Iterate through the large_boxes
            self.randomly_place_obstacle(f'large_box_{obst_num_counter}', [random_positions[idx_counter][0],random_positions[idx_counter][1],0], random_positions[idx_counter][2])
            idx_counter += 1
            #Iterate through the small cylinders
            self.randomly_place_obstacle(f'small_cylinder_{obst_num_counter}', [random_positions[idx_counter][0],random_positions[idx_counter][1],0], random_positions[idx_counter][2])
            idx_counter += 1
            #Iterate through the medium cylinders
            self.randomly_place_obstacle(f'medium_cylinder_{obst_num_counter}', [random_positions[idx_counter][0],random_positions[idx_counter][1],0], random_positions[idx_counter][2])
            idx_counter += 1
            #Iterate through the large cylinders
            self.randomly_place_obstacle(f'large_cylinder_{obst_num_counter}', [random_positions[idx_counter][0],random_positions[idx_counter][1],0], random_positions[idx_counter][2])
            idx_counter += 1

            #Next obstacle number
            obst_num_counter +=1

        #Iterate through all obstacles to move them
        #Unpause physics
        self.unpause_physics_proxy(EmptyRequest())
        return (1,'Succesfully scrambled environment')

if __name__ == "__main__":
    rospy.init_node('world_randomizer')
    EnvScrambler()
    rospy.loginfo("Succesfully made the service")
    rospy.spin()
    