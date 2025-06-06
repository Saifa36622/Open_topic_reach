from pyURControl import ur_control
from time import sleep
from operator import add
import math
# from rtde_control 
'''
make your program here
to Install pyURControl localy, run the following command: pip install -e .
'''
def program():
    # Init UR Control
    ur_control.init('192.168.56.101')

    # Send power on command
    ur_control.power_on()

    # Send break release command
    ur_control.break_release()

    # Do pick and place
    PickAndPlace()

    # Read inputs - Test on UR-sim
    # while True:
    #     # read digital inputs
    #     print("Inputs: ", ur_control.get_digital_inputs())
    #     sleep(0.01)
    

def PickAndPlace():
    # set tcp
    # ur_control.set_tcp([0, 0, 0.2, 0, 0, 0])

    # set payload if there is any
    # ur_control.set_payload(0, [0, 0, 0], [0,0,0,0,0,0])

    # define positions
    # p1_home = [0.0, 0.5, 0.1, 0.0, 0.0, 0.0]
    p1 = [1.57, -math.pi/2, 0, 0, math.pi/2, 0.0]
    p2 = [0, -math.pi/2, -1.57, 0.0, math.pi/2, 0.0]

    # Pick 

    # print("begin move")
    check2 = ur_control.move_directly(p1, a=2.4, v=2.05, t=0, r=0.02) # move to p1_home
    check1 = ur_control.move_directly(p2, a=1.4, v=1.05, t=0, r=0.02) # move to p1_home
    # print("done move")
    # ur_control.move_joint_with_pose(list(map(add,p2_pick,p2_offset_approach)), a=1.4, v=1.05, t=0, r=0) # move to p2_pick with offset for approach
    # ur_control.move_linear_pose(p2_pick, a=1.4, v=1.05, t=0, r=0) # move to p2_pick with linear motion
    # # pick the piece - close gripper
    # CloseGripper()
    # ##
    # ur_control.move_joint_with_pose(list(map(add,p2_pick,p2_offset_leave)), a=1.4, v=1.05, t=0, r=0) # move to p2_pick with offset to leave

    # # Place
    # ur_control.move_joint_with_pose(list(map(add,p3_place,p3_offset_approach)), a=1.4, v=1.05, t=0, r=0) # move to p3_place with offset for approach
    # ur_control.move_linear_pose(p3_place, a=1.4, v=1.05, t=0, r=0) # move to p3_place with linear motion
    # # place the piece - open gripper
    # OpenGripper()
    # ##
    # ur_control.move_joint_with_pose(list(map(add,p3_place,p3_offset_leave)), a=1.4, v=1.05, t=0, r=0) # move to p3_place with offset to leave

def OpenGripper():
    ur_control.set_digital_output(0, False)
    ur_control.set_digital_output(4, True)

def CloseGripper():
    ur_control.set_digital_output(0, True)
    ur_control.set_digital_output(4, False)


if __name__ == '__main__':
    program()