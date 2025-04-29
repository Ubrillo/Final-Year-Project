#######################################################################
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import ros2_numpy as rnp
from ultralytics import YOLO
from geometry_msgs.msg import Twist
import cv2

class DetectionNode(Node):
    def __init__(self):
        super().__init__("ultralytics")

        #path to model
        model_path = "/home/grandson/ros2_ws/src/object_detection/object_detection/best-latest.pt"

        #loads the YOLO model
        self.detection_model = YOLO(model_path)
        
        # Publisher for detected object
        self.detection_pub = self.create_publisher(Image, "/ultralytics/detection/classes", 10)
        
        # Publisher for detected classes size
        self.size_pub = self.create_publisher(String, "/ultralytics/size/classes", 10)

        #velocity publisher
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriber for the camera image
        self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.perception, 10)

        # Sleep for a second to ensure everything is initialized
        time.sleep(1)
        # self.signal_goal = 0
        self.robot_state = ""
        self.turn_signal = 0
        self.avoid_distance = 0.75
        self.step_time =  10
        self.start_time = 0.0
        self.dive_signal = 0
        self.collision_height = 0
        self.goal_signal = False
        
    #perception Layer
    def perception(self, img_data):
        """Callback function to process image and publish detected class names."""
        
        #converts ros data to numpy
        array = rnp.numpify(img_data)

        #check if a node  has subcribed to the topic
        if self.detection_pub.get_subscription_count():
            #run an inference
            self.results = self.detection_model(array, conf=0.7)
            
            #plot bounding box on frame
            annotated_frame = self.results[0].plot(show=False)
            
            #converts the annotated frame to ros Image data fromat and publish
            self.detection_pub.publish(rnp.msgify(Image, annotated_frame, encoding='bgr8'))
            
            #extracts  class in a frame as integers
            classes = self.results[0].boxes.cls.cpu().numpy().astype(int)
            
            #extracts class name in a frame name
            self.names = [self.results[0].names[i] for i in classes]
            
            #loops through each bounding box and class name
            for box, name in zip(self.results[0].boxes, self.names):

                #extracts the dimensions and location of bounding box
                ox_center, oy_center, width, height = map(int, box.xyxy[0].cpu().numpy())
                str_width = str(width)
                str_height = str(height)
                str_size = str(width * height)

                #calcuate the distance between the center of a box and the center of an image
                diff = 320 - (ox_center+width//2)
                str_diff = str(diff)
                text = "size: "+str_size + " height: "+str_height  +" width: "+str_width+ " diff: "+str_diff

                #publish class size
                self.size_pub.publish(String(data = (text)))
                self.collision_height = height
                self.get_logger().info(str(name))    

                #checks if object is a cone and print the location of the cone
                if self.isCone(name):
                     self.get_logger().info(str(ox_center))
                
                #activate the control function with object dimensions
                self.control(name, height, ox_center)

            #checks if no detection is found and activate control function with empty parameters
            if len(self.results) == 0 or self.results == None or len(self.results[0].boxes) == 0:
                self.control()

    #sends velocity commands to the robot
    def control(self, obs_name=None,  obs_height=None, ox_center=None):
        #validates the dimension of an object
        if obs_name and obs_height and ox_center:
            #validate a goal in a frame
            if self.seeGoal(self.names) or self.goal_signal:
                self.goal_signal = True
                if self.isCone(obs_name): #validate a cone in a frame
                    if obs_height >= 230: #validate collision
                        self.turn_signal = 1 
                        self.dive_signal = 0
                        self.get_logger().info("Avoiding obstacle")
                        self.avoid_obstacle(ox_center) #actiavtes the robot to avoid obstacle
                        self.goal_signal = False
                    else:
                        self.get_logger().info("I'm moving forward")
                        self.drive_forward() #activates robot to move forward
                else:
                    self.drive_forward() #activates robot to move forward
        else:
            if self.turn_signal == 1:
                self.turn_signal = 0
                time.sleep(2)
                self.start_time = time.time()
                elapsed_time = 0.0
                # self.step_time = (self.collision_height * 15) / 230 
                self.get_logger().info("I'm diving for "+str(self.step_time)+" seconds...")
                self.get_logger().info("collision height: "+str(self.collision_height))

                #caveat state
                while elapsed_time < self.step_time:
                    self.drive_forward()
                    elapsed_time = time.time() - self.start_time
                self.dive_signal = 1
            
            if self.dive_signal == 1:
                self.dive_signal = 2 
            if self.dive_signal == 2:
                self.goal_signal=False
                self.searchGoal()
            
            # if self.turn_signal == 1:
            #     if self.seeGoal(self.names) == False:
            #         if self.isCone(obs_name):
            #             if self.collision_height()
            #             self.avoid_obstacle(ox_center)

            self.get_logger().info("turn signal: "+ str(self.turn_signal))
        
        
    def plot(self):
        
        # Suppose you have these bounding box coordinates
        x1, y1, x2, y2 = 100, 50, 200, 150  # top-left and bottom-right
        # Text to display (e.g., class label)
        label = "TL"

        # Optional: font and styling
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)        # white text

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Optional: draw background rectangle behind text
        cv2.rectangle(image, (x1, y1 - text_height - baseline), (x1 + text_width, y1), bg_color, -1)

        # Draw the text
        cv2.putText(array, label, (x1, y1 - baseline), font, font_scale, color, thickness)


    #moves the robot forward
    def drive_forward(self, speed = 0.09):
        msg = Twist()
        msg.linear.x = speed
        msg.angular.z = 0.0
        self.cmd_publisher.publish(msg)
    
    #turn left or right to avoid obstacle
    def avoid_obstacle(self, msg):
        """Stops and turns the robot to avoid an obstacle."""  
        twist_msg = Twist()
        if msg >= 125:
            self.get_logger().info("turning left "+str(msg))
            self.robot_state = 'left-turn'
            twist_msg.angular.z = 0.1
            twist_msg.linear.x = 0.0
            self.cmd_publisher.publish(twist_msg)     
            time.sleep(0.5)
            
        elif msg  < 125:
            self.get_logger().info("turning right "+str(msg))
            twist_msg.angular.z = -0.1
            twist_msg.linear.x = 0.0
            self.robot_state = 'right-turn'
            self.cmd_publisher.publish(twist_msg)
            time.sleep(0.5)

    #turn left or right to search for goal
    def searchGoal(self):
        twist_msg = Twist()
        if self.robot_state == 'right-turn':
            twist_msg.angular.z = 0.05
            twist_msg.linear.x = 0.0
            self.cmd_publisher.publish(twist_msg)
            self.get_logger().info("searching turning left")
            time.sleep(0.5)
        elif self.robot_state == 'left-turn':
            twist_msg.angular.z = -0.05
            twist_msg.linear.x = 0.0
            self.cmd_publisher.publish(twist_msg)
            self.get_logger().info("searching - turning right")
            time.sleep(0.5)
    
    #return True if cone is present in an image
    def isCone(self, name):
        cones = ['cone_blue', 'cone_red', 'cone_yellow', 'cone_green', 'cone_purple']
        if name in cones:
            return True
        return False
   
   #return True if cone is detected false otherwise
    def checkCones(self, names):
        for i in names:
            if self.isCone(i):
                return True
        return False
    
    #return True if Goal is detected false otherwise
    def seeGoal(self, names):
        for i in names:
            if i == 'goal':
                return True
        return False
    
    #return True if collision is detected false otherwise
    def isCollision(self, height, max_height=230):
        if height >= max_height:
            return True
        return False
        
def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()