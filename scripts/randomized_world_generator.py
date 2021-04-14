import numpy as np
import random

x_range = [0,56.54]
y_range = [-4,4]
num_obstacles = 15

random_positions = np.array([np.random.uniform([x_range[0],y_range[0]],[x_range[1],y_range[1]]) for x in range(num_obstacles)])


obstacles = ["rock_formation_1","rock_formation_2","rock_formation_3","rock_formation_4","simple_pyramid"]
#grey_wall_string = "\t\t<include> \n\t\t <name>wall_1</name> <uri>model://grey_wall</uri> <pose>56.54 0.02 0.000000 0 0 0.000000</pose> </include>"
counter = 0
for i in range(num_obstacles):
    rand_obs = random.choice(obstacles)
    if rand_obs == "rock_formation_1":
        print(f"\t<include> \n\t <name>wall_{counter}</name>\n\t\t <uri>model://Simple Stone</uri>\n\t\t <pose>{random_positions[i][0]-0.89} {random_positions[i][1]-0.32} 0.000000 0 0 0.000000</pose> \n\t</include>")
        counter +=1
        print(f"\t<include> \n\t <name>wall_{counter}</name>\n\t\t <uri>model://Simple Stone</uri>\n\t\t <pose>{random_positions[i][0]-0.63} {random_positions[i][1]+0.46} 0.000000 0 0 0.000000</pose> \n\t</include>")
        counter +=1
        print(f"\t<include> \n\t <name>wall_{counter}</name>\n\t\t <uri>model://Simple Stone</uri>\n\t\t <pose>{random_positions[i][0]+0.139} {random_positions[i][1]+0.837} 0.000000 0 0 0.000000</pose> \n\t</include>")
        counter +=1
        print(f"\t<include> \n\t <name>wall_{counter}</name>\n\t\t <uri>model://Simple Stone</uri>\n\t\t <pose>{random_positions[i][0]+0.886} {random_positions[i][1]+0.756} 0.000000 0 0 0.000000</pose> \n\t</include>")
        counter +=1
        print(f"\t<include> \n\t <name>wall_{counter}</name>\n\t\t <uri>model://Simple Stone</uri>\n\t\t <pose>{random_positions[i][0]+0.8} {random_positions[i][1]-0.33} 0.000000 0 0 0.000000</pose> \n\t</include>")
        counter +=1
        print(f"\t<include> \n\t <name>wall_{counter}</name>\n\t\t <uri>model://Simple Stone</uri>\n\t\t <pose>{random_positions[i][0]-0.039} {random_positions[i][1]-0.65} 0.000000 0 0 0.000000</pose> \n\t</include>")
        counter +=1

    if rand_obs == "rock_formation_2":
        print(f"\t<include> \n\t <name>wall_{counter}</name>\n\t\t <uri>model://Simple Stone</uri>\n\t\t <pose>{random_positions[i][0]-0.4} {random_positions[i][1]-0.62} 0.000000 0 0 0.000000</pose> \n\t</include>")
        counter +=1
        print(f"\t<include> \n\t <name>wall_{counter}</name>\n\t\t <uri>model://Simple Stone</uri>\n\t\t <pose>{random_positions[i][0]-0.389} {random_positions[i][1]+0.234} 0.000000 0 0 0.000000</pose> \n\t</include>")
        counter +=1
        print(f"\t<include> \n\t <name>wall_{counter}</name>\n\t\t <uri>model://Simple Stone</uri>\n\t\t <pose>{random_positions[i][0]-0.348} {random_positions[i][1]+1.165} 0.000000 0 0 0.000000</pose> \n\t</include>")
        counter +=1
        print(f"\t<include> \n\t <name>wall_{counter}</name>\n\t\t <uri>model://Simple Stone</uri>\n\t\t <pose>{random_positions[i][0]+0.34} {random_positions[i][1]+1.14} 0.000000 0 0 0.000000</pose> \n\t</include>")
        counter +=1
        print(f"\t<include> \n\t <name>wall_{counter}</name>\n\t\t <uri>model://Simple Stone</uri>\n\t\t <pose>{random_positions[i][0]+0.3} {random_positions[i][1]-0.29} 0.000000 0 0 0.000000</pose> \n\t</include>")
        counter +=1
        print(f"\t<include> \n\t <name>wall_{counter}</name>\n\t\t <uri>model://Simple Stone</uri>\n\t\t <pose>{random_positions[i][0]+0.345} {random_positions[i][1]-0.61} 0.000000 0 0 0.000000</pose> \n\t</include>")
        counter +=1
    if rand_obs == "rock_formation_3":
        print(f"\t<include> \n\t <name>wall_{counter}</name>\n\t\t <uri>model://Simple Stone</uri>\n\t\t <pose>{random_positions[i][0]-0.34} {random_positions[i][1]-0.55} 0.000000 0 0 0.000000</pose> \n\t</include>")
        counter +=1
        print(f"\t<include> \n\t <name>wall_{counter}</name>\n\t\t <uri>model://Simple Stone</uri>\n\t\t <pose>{random_positions[i][0]-0.29} {random_positions[i][1]+0.46} 0.000000 0 0 0.000000</pose> \n\t</include>")
        counter +=1
        print(f"\t<include> \n\t <name>wall_{counter}</name>\n\t\t <uri>model://Simple Stone</uri>\n\t\t <pose>{random_positions[i][0]+0.16} {random_positions[i][1]+1.29} 0.000000 0 0 0.000000</pose> \n\t</include>")
        counter +=1
        print(f"\t<include> \n\t <name>wall_{counter}</name>\n\t\t <uri>model://Simple Stone</uri>\n\t\t <pose>{random_positions[i][0]+0.866} {random_positions[i][1]+1.69} 0.000000 0 0 0.000000</pose> \n\t</include>")
        counter +=1
        print(f"\t<include> \n\t <name>wall_{counter}</name>\n\t\t <uri>model://Simple Stone</uri>\n\t\t <pose>{random_positions[i][0]+0.74} {random_positions[i][1]+0.7} 0.000000 0 0 0.000000</pose> \n\t</include>")
        counter +=1
        print(f"\t<include> \n\t <name>wall_{counter}</name>\n\t\t <uri>model://Simple Stone</uri>\n\t\t <pose>{random_positions[i][0]+0.44} {random_positions[i][1]-0.86} 0.000000 0 0 0.000000</pose> \n\t</include>")
        counter +=1
    if rand_obs == "rock_formation_4":
        print(f"\t<include> \n\t <name>wall_{counter}</name>\n\t\t <uri>model://Simple Stone</uri>\n\t\t <pose>{random_positions[i][0]-01.39} {random_positions[i][1]+0.05} 0.000000 0 0 0.000000</pose> \n\t</include>")
        counter +=1
        print(f"\t<include> \n\t <name>wall_{counter}</name>\n\t\t <uri>model://Simple Stone</uri>\n\t\t <pose>{random_positions[i][0]-0.38} {random_positions[i][1]+0.85} 0.000000 0 0 0.000000</pose> \n\t</include>")
        counter +=1
        print(f"\t<include> \n\t <name>wall_{counter}</name>\n\t\t <uri>model://Simple Stone</uri>\n\t\t <pose>{random_positions[i][0]-0.426} {random_positions[i][1]} 0.000000 0 0 0.000000</pose> \n\t</include>")
        counter +=1
        print(f"\t<include> \n\t <name>wall_{counter}</name>\n\t\t <uri>model://Simple Stone</uri>\n\t\t <pose>{random_positions[i][0]+0.594} {random_positions[i][1]} 0.000000 0 0 0.000000</pose> \n\t</include>")
        counter +=1
        print(f"\t<include> \n\t <name>wall_{counter}</name>\n\t\t <uri>model://Simple Stone</uri>\n\t\t <pose>{random_positions[i][0]-0.42} {random_positions[i][1]-0.877} 0.000000 0 0 0.000000</pose> \n\t</include>")
        counter +=1
    if rand_obs == "simple_pyramid":
        print(f"\t<include> \n\t <name>pyramid_{counter}</name>\n\t\t <uri>model://Simple Pyramid</uri>\n\t\t <pose>{random_positions[i][0]} {random_positions[i][1]} 0.000000 0 0 0.000000</pose> \n\t</include>")
        counter +=1


