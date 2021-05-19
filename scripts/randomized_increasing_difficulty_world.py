import numpy as np
import random

x_range = [-50,50]
y_range = [-4,4]
z_range = [0,10] #randomize height of objects. Objects will be half this size, as center is placed at z = 0
z_rot_range = [0,6.28]
num_obstacles = 3

f = open("obstacles.txt", "w")


#random_positions = np.array([np.random.uniform([x_range[0],y_range[0],z_range[0],z_rot_range[0]],[x_range[1],y_range[1],z_range[1],z_rot_range[1]]) for x in range(num_obstacles)])
dx = 6
dy = 3

#Large box
counter = 0
for i in range(num_obstacles):
    f.write(f"<model name='large_box_{counter}'>\n\t<static>1</static>\n\t<pose frame=''> {(counter%3)*dx} 8 1.25 0 -0 0</pose>\n\t\t<link name='link'>\n\t\t\t<inertial>\n\t\t\t\t<mass>1</mass>\n\t\t\t\t<inertia>\n\t\t\t\t\t<ixx>0.1</ixx>\n\t\t\t\t\t<ixy>0</ixy>\n\t\t\t\t\t<ixz>0</ixz>\n\t\t\t\t\t<iyy>0.1</iyy>\n\t\t\t\t\t<iyz>0</iyz>\n\t\t\t\t\t<izz>0.1</izz>\n\t\t\t\t</inertia>\n\t\t\t<pose frame=''>0 0 0 0 -0 0</pose>\n\t\t\t</inertial>\n\t\t<collision name='collision'>\n\t\t\t<geometry>\n\t\t\t\t<box>\n\t\t\t\t\t<size> 2.5 2.5 3 </size>\n\t\t\t\t</box>\n\t\t\t</geometry>\n\t\t<max_contacts>10</max_contacts>\n\t\t<surface>\n\t\t\t<contact>\n\t\t\t\t<ode/>\n\t\t\t</contact>\n\t\t<bounce/>\n\t\t<friction>\n\t\t\t<torsional>\n\t\t\t\t<ode/>\n\t\t\t</torsional>\n\t\t<ode/>\n\t</friction>\n</surface>\n</collision>\n<visual name='visual'>\n\t\t<geometry>\n\t\t\t<box>\n\t\t\t\t<size> 2.5 2.5 3 </size>\n\t\t\t</box>\n\t\t</geometry>\n\t\t<material>\n\t\t\t<script>\n\t\t\t\t<name>Gazebo/Grey</name>\n\t\t\t\t<uri>file://media/materials/scripts/gazebo.material</uri>\n\t\t\t</script>\n\t\t</material>\n\t</visual>\n<self_collide>0</self_collide>\n<enable_wind>0</enable_wind>\n<kinematic>0</kinematic>\n</link>\n</model>")
    counter +=1

#Medium box
counter = 0
for i in range(num_obstacles):
    f.write(f"<model name='medium_box_{counter}'>\n\t<static>1</static>\n\t<pose frame=''>{(counter%3)*dx} 12 1 0 -0 0</pose>\n\t\t<link name='link'>\n\t\t\t<inertial>\n\t\t\t\t<mass>1</mass>\n\t\t\t\t<inertia>\n\t\t\t\t\t<ixx>0.1</ixx>\n\t\t\t\t\t<ixy>0</ixy>\n\t\t\t\t\t<ixz>0</ixz>\n\t\t\t\t\t<iyy>0.1</iyy>\n\t\t\t\t\t<iyz>0</iyz>\n\t\t\t\t\t<izz>0.1</izz>\n\t\t\t\t</inertia>\n\t\t\t<pose frame=''>0 0 0 0 -0 0</pose>\n\t\t\t</inertial>\n\t\t<collision name='collision'>\n\t\t\t<geometry>\n\t\t\t\t<box>\n\t\t\t\t\t<size> 1.5 1.5 1.5 </size>\n\t\t\t\t</box>\n\t\t\t</geometry>\n\t\t<max_contacts>10</max_contacts>\n\t\t<surface>\n\t\t\t<contact>\n\t\t\t\t<ode/>\n\t\t\t</contact>\n\t\t<bounce/>\n\t\t<friction>\n\t\t\t<torsional>\n\t\t\t\t<ode/>\n\t\t\t</torsional>\n\t\t<ode/>\n\t</friction>\n</surface>\n</collision>\n<visual name='visual'>\n\t\t<geometry>\n\t\t\t<box>\n\t\t\t\t<size> 2 2 2 </size>\n\t\t\t</box>\n\t\t</geometry>\n\t\t<material>\n\t\t\t<script>\n\t\t\t\t<name>Gazebo/Grey</name>\n\t\t\t\t<uri>file://media/materials/scripts/gazebo.material</uri>\n\t\t\t</script>\n\t\t</material>\n\t</visual>\n<self_collide>0</self_collide>\n<enable_wind>0</enable_wind>\n<kinematic>0</kinematic>\n</link>\n</model>")
    counter +=1

# #Large sphere
# counter = 0
# for i in range(num_obstacles):
#     f.write(f"<model name='large_sphere_{counter}'>\n\t<static>1</static>\n\t<pose frame=''>{(counter%3)*dx} 16 1.5 0 -0 0</pose>\n\t\t<link name='link'>\n\t\t\t<inertial>\n\t\t\t\t<mass>1</mass>\n\t\t\t\t<inertia>\n\t\t\t\t\t<ixx>0.1</ixx>\n\t\t\t\t\t<ixy>0</ixy>\n\t\t\t\t\t<ixz>0</ixz>\n\t\t\t\t\t<iyy>0.1</iyy>\n\t\t\t\t\t<iyz>0</iyz>\n\t\t\t\t\t<izz>0.1</izz>\n\t\t\t\t</inertia>\n\t\t\t<pose frame=''>0 0 0 0 -0 0</pose>\n\t\t\t</inertial>\n\t\t<collision name='collision'>\n\t\t\t<geometry>\n\t\t\t\t<sphere>\n\t\t\t\t\t<radius> 3 </radius>\n\t\t\t\t</sphere>\n\t\t\t</geometry>\n\t\t<max_contacts>10</max_contacts>\n\t\t<surface>\n\t\t\t<contact>\n\t\t\t\t<ode/>\n\t\t\t</contact>\n\t\t<bounce/>\n\t\t<friction>\n\t\t\t<torsional>\n\t\t\t\t<ode/>\n\t\t\t</torsional>\n\t\t<ode/>\n\t</friction>\n</surface>\n</collision>\n<visual name='visual'>\n\t\t<geometry>\n\t\t\t<sphere>\n\t\t\t\t<radius> 3 </radius>\n\t\t\t</sphere>\n\t\t</geometry>\n\t\t<material>\n\t\t\t<script>\n\t\t\t\t<name>Gazebo/Grey</name>\n\t\t\t\t<uri>file://media/materials/scripts/gazebo.material</uri>\n\t\t\t</script>\n\t\t</material>\n\t</visual>\n<self_collide>0</self_collide>\n<enable_wind>0</enable_wind>\n<kinematic>0</kinematic>\n</link>\n</model>")
#     counter +=1

# #Medium sphere
# counter = 0
# for i in range(num_obstacles):
#     f.write(f"<model name='medium_sphere_{counter}'>\n\t<static>1</static>\n\t<pose frame=''>{(counter%3)*dx} 20 1 0 -0 0</pose>\n\t\t<link name='link'>\n\t\t\t<inertial>\n\t\t\t\t<mass>1</mass>\n\t\t\t\t<inertia>\n\t\t\t\t\t<ixx>0.1</ixx>\n\t\t\t\t\t<ixy>0</ixy>\n\t\t\t\t\t<ixz>0</ixz>\n\t\t\t\t\t<iyy>0.1</iyy>\n\t\t\t\t\t<iyz>0</iyz>\n\t\t\t\t\t<izz>0.1</izz>\n\t\t\t\t</inertia>\n\t\t\t<pose frame=''>0 0 0 0 -0 0</pose>\n\t\t\t</inertial>\n\t\t<collision name='collision'>\n\t\t\t<geometry>\n\t\t\t\t<sphere>\n\t\t\t\t\t<radius> 1.5 </radius>\n\t\t\t\t</sphere>\n\t\t\t</geometry>\n\t\t<max_contacts>10</max_contacts>\n\t\t<surface>\n\t\t\t<contact>\n\t\t\t\t<ode/>\n\t\t\t</contact>\n\t\t<bounce/>\n\t\t<friction>\n\t\t\t<torsional>\n\t\t\t\t<ode/>\n\t\t\t</torsional>\n\t\t<ode/>\n\t</friction>\n</surface>\n</collision>\n<visual name='visual'>\n\t\t<geometry>\n\t\t\t<sphere>\n\t\t\t\t<radius> 1.5 </radius>\n\t\t\t</sphere>\n\t\t</geometry>\n\t\t<material>\n\t\t\t<script>\n\t\t\t\t<name>Gazebo/Grey</name>\n\t\t\t\t<uri>file://media/materials/scripts/gazebo.material</uri>\n\t\t\t</script>\n\t\t</material>\n\t</visual>\n<self_collide>0</self_collide>\n<enable_wind>0</enable_wind>\n<kinematic>0</kinematic>\n</link>\n</model>")
#     counter +=1

#Large cylinder
counter = 0
for i in range(num_obstacles):
    f.write(f"<model name='large_cylinder_{counter}'>\n\t<static>1</static>\n\t<pose frame=''>{(counter%3)*dx} 24 2.5 0 -0 0</pose>\n\t\t<link name='link'>\n\t\t\t<inertial>\n\t\t\t\t<mass>1</mass>\n\t\t\t\t<inertia>\n\t\t\t\t\t<ixx>0.1</ixx>\n\t\t\t\t\t<ixy>0</ixy>\n\t\t\t\t\t<ixz>0</ixz>\n\t\t\t\t\t<iyy>0.1</iyy>\n\t\t\t\t\t<iyz>0</iyz>\n\t\t\t\t\t<izz>0.1</izz>\n\t\t\t\t</inertia>\n\t\t\t<pose frame=''>0 0 0 0 -0 0</pose>\n\t\t\t</inertial>\n\t\t<collision name='collision'>\n\t\t\t<geometry>\n\t\t\t\t<cylinder>\n\t\t\t\t\t<radius> 1.5 </radius>\n\t\t\t\t<length> 5 </length>\n\t\t\t\t</cylinder>\n\t\t\t</geometry>\n\t\t<max_contacts>10</max_contacts>\n\t\t<surface>\n\t\t\t<contact>\n\t\t\t\t<ode/>\n\t\t\t</contact>\n\t\t<bounce/>\n\t\t<friction>\n\t\t\t<torsional>\n\t\t\t\t<ode/>\n\t\t\t</torsional>\n\t\t<ode/>\n\t</friction>\n</surface>\n</collision>\n<visual name='visual'>\n\t\t<geometry>\n\t\t\t<cylinder>\n\t\t\t\t<radius> 1.5 </radius>\n\t\t\t<length> 5 </length> \n\t\t\t\t</cylinder>\n\t\t</geometry>\n\t\t<material>\n\t\t\t<script>\n\t\t\t\t<name>Gazebo/Grey</name>\n\t\t\t\t<uri>file://media/materials/scripts/gazebo.material</uri>\n\t\t\t</script>\n\t\t</material>\n\t</visual>\n<self_collide>0</self_collide>\n<enable_wind>0</enable_wind>\n<kinematic>0</kinematic>\n</link>\n</model>")
    counter +=1

#Medium cylinder
counter = 0
for i in range(num_obstacles):
    f.write(f"<model name='medium_cylinder_{counter}'>\n\t<static>1</static>\n\t<pose frame=''>{(counter%3)*dx} 28 1.5 0 -0 0</pose>\n\t\t<link name='link'>\n\t\t\t<inertial>\n\t\t\t\t<mass>1</mass>\n\t\t\t\t<inertia>\n\t\t\t\t\t<ixx>0.1</ixx>\n\t\t\t\t\t<ixy>0</ixy>\n\t\t\t\t\t<ixz>0</ixz>\n\t\t\t\t\t<iyy>0.1</iyy>\n\t\t\t\t\t<iyz>0</iyz>\n\t\t\t\t\t<izz>0.1</izz>\n\t\t\t\t</inertia>\n\t\t\t<pose frame=''>0 0 0 0 -0 0</pose>\n\t\t\t</inertial>\n\t\t<collision name='collision'>\n\t\t\t<geometry>\n\t\t\t\t<cylinder>\n\t\t\t\t\t<radius> 1 </radius>\n\t\t\t\t<length> 3 </length>\n\t\t\t\t</cylinder>\n\t\t\t</geometry>\n\t\t<max_contacts>10</max_contacts>\n\t\t<surface>\n\t\t\t<contact>\n\t\t\t\t<ode/>\n\t\t\t</contact>\n\t\t<bounce/>\n\t\t<friction>\n\t\t\t<torsional>\n\t\t\t\t<ode/>\n\t\t\t</torsional>\n\t\t<ode/>\n\t</friction>\n</surface>\n</collision>\n<visual name='visual'>\n\t\t<geometry>\n\t\t\t<cylinder>\n\t\t\t\t<radius> 1 </radius>\n\t\t\t<length> 3 </length> \n\t\t\t\t</cylinder>\n\t\t</geometry>\n\t\t<material>\n\t\t\t<script>\n\t\t\t\t<name>Gazebo/Grey</name>\n\t\t\t\t<uri>file://media/materials/scripts/gazebo.material</uri>\n\t\t\t</script>\n\t\t</material>\n\t</visual>\n<self_collide>0</self_collide>\n<enable_wind>0</enable_wind>\n<kinematic>0</kinematic>\n</link>\n</model>")
    counter +=1

f.close()



