#!/usr/bin/env python
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from scipy.spatial import KDTree # KDTree is a data structure that allows to look up the closest point in space really efficiently
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

Notes: KDTree - This class provides an index into a set of k-dimensional points which can be used to rapidly look up the nearest neighbors of any point.
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish (look ahead) ahead of the car
MAX_DECEL = 0.5 # Max Deceleration

class WaypointUpdater(object):
    
    def __init__(self):
        rospy.init_node('waypoint_updater')
        
        # Member variables
        self.base_lane = None
        self.prev_state = self.now_state = -1
        self.pose = None
        self.stopline_wp_idx = -1
        self.waypoints_2d = None
        self.waypoint_tree = None

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb) # current position of the vehicle to be used to locate waypoints ahead of car
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb) # list of waypoints ahead/behind of vehicle using waypoints_cb callback method, these do not change
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb) # taking into account the locations to stop for red traffic lights
        # rospy.Subscriber('/obstacle_waypoint', Waypoint, self.obstacle_cb) # taking into account the locations of obstacles

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        
        self.loop() # this gives us control about the publishing frequency
    
    def loop(self):
        rate = rospy.Rate(30) # 30 Hertz, Autoware is running at 30Hz
        while not rospy.is_shutdown():
            if self.pose and self.base_lane:
                # Get closest waypoint
                self.publish_waypoints()
            rate.sleep()
            
    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x # x position of vehicle from msg definitions
        y = self.pose.pose.position.y # y position of vehicle from msg definitions
        # Perform query on KDTree based on x and y.  query([x,y], 1)[1] - The first 1 is we only want to return 1 item from the KDTree, so the closest point
        # in our KDTree to our query item.  The second [1] will return the position and also the index which is in the same order as we put into the KDTree
        closest_idx = self.waypoint_tree.query([x,y],1)[1]
        
        # Check if closest waypoint is ahead or behind the vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord) # closest coordinate vector
        prev_vect = np.array(prev_coord) # previous coordinate vector
        pos_vect = np.array([x,y]) # position vector
        
        # dot product to see if waypoint is positive or negative between two vectors cl_vect and prev_vect
        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)
        
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx
    
    def publish_waypoints(self):

        final_lane = self.generate_lane()
        
        self.final_waypoints_pub.publish(final_lane)
        
    def generate_lane (self):
        lane = Lane()
        
        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]

        # If stopline_wp_idx is -1 or further way they we care about, we will publish the waypoints directly
        if self.stopline_wp_idx == -1 or self.stopline_wp_idx >= farthest_idx:
            lane.waypoints = base_waypoints
        # We detected a redlight and 
        else:
            # temp is a newly created list of waypoints for our lane that is returned from decelerate_waypoints 
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)
        # return our new lane to publish_waypoints
        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = [] # create a new list of waypoints using information from base_waypoints
        # first enumerate over the sliced list of waypoints 
        for i, wp in enumerate(waypoints):
            # set the pose to the base waypoint pose
            p = Waypoint()
            p.pose = wp.pose
            
            # Need to determine the stop index
            stop_idx = max(self.stopline_wp_idx - closest_idx - 3 , 0)  # -3 is to ensure nose of the car is behind the 
            # stopline as its based on the centre of the car.
            
            # distance function sums up the line segements between all the waypoints
            dist = self.distance(waypoints, i, stop_idx)
            # velocity follows a sqrt function, as distance to waypoint becomes closer to waypoint the speed slows.
            vel = math.sqrt(2 * MAX_DECEL * dist)

            if vel < 1.:
                vel = 0
                
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)
            
        return temp
    
    def pose_cb(self, msg):
        # 'cb' for callback
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # TODO: Implement;  This is called only once since the waypoints do not change.  This is a latched subscriber
        self.base_lane = waypoints
        # if not self.waypoints_2d is used because that attribute is being used in the callback here, want to make sure 
        # that self.waypoints_2d is initialized        
        # before # the subscriber is, other wise you could run into some risk conditions where the subscriber callback is         # called before waypoints_2d is 
        # initialized.
        if not self.waypoints_2d:
            # For each waypoint convert to just the 2D coordinates for each waypoint.
            # We have for each waypoint and waypoints.waypoints we have waypoint.pose.pose.position.x and 
            # waypoint.pose.pose.position.y
            # Will give us a list of a bunch of 2D coordinates for each waypoint
            # We use this list to construct a KDTree
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
                                 
    def traffic_cb(self, msg):
        # Callback for /traffic_waypoint message
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')