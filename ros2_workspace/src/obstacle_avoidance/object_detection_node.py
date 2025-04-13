######################################################################
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import ros2_numpy as rnp
from ultralytics import YOLO
from geometry_msgs.msg import Twist

class DetectionNode(Node):
    def __init__(self):
        super().__init__("ultralytics")

        model_path = "/home/grandson/ros2_ws/src/object_detection/object_detection/best_keep.pt"
        self.detection_model = YOLO(model_path)
        
        # Publisher for detected classes
        self.detection_pub = self.create_publisher(Image, "/ultralytics/detection/classes", 10)
        
        # Publisher for detected classes size
        self.size_pub = self.create_publisher(String, "/ultralytics/size/classes", 10)

        #velocity publisher
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriber for the camera image
        self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.callback, 10)

        # Sleep for a second to ensure everything is initialized
        time.sleep(1)
        self.signal_goal = 0
        self.robot_state = ""
        self.turn_signal = 0
        self.avoid_distance = 0.75
        self.step_time =  0.75/0.05
        self.start_time = 0.0
        self.dive_signal = 0
        self.collision_height = 0
        

    def callback(self, img_data):
        """Callback function to process image and publish detected class names."""
        array = rnp.numpify(img_data)

        if self.detection_pub.get_subscription_count():
    
            results = self.detection_model(array, conf=0.7)
            annotated_frame = results[0].plot(show=False)
            self.detection_pub.publish(rnp.msgify(Image, annotated_frame, encoding='bgr8'))
            
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            names = [results[0].names[i] for i in classes]
            
            for box, name in zip(results[0].boxes, names):
                ox_center, oy_center, width, height = map(int, box.xyxy[0].cpu().numpy())
                str_width = str(width)
                str_height = str(height)
                str_size = str(width * height)
                diff = 320 - (ox_center+width//2)
                str_diff = str(diff)
                text = "size: "+str_size + " height: "+str_height  +" width: "+str_width+ " diff: "+str_diff
                self.size_pub.publish(String(data = (text)))
                self.collision_height = height
                
                self.get_logger().info(str(name))            
                if self.isCone(name):
                     self.get_logger().info(str(ox_center))
                    

                # self.get_logger().info(str(ox_center))
                if self.seeGoal(names):
                    if self.isCone(name):
                        if height >= 230:
                            self.turn_signal = 1
                            self.dive_signal = 0
                            self.avoid_obstacle(ox_center)
                        else:
                            self.drive_forward()
                    else:
                        self.drive_forward()
          
                           
            self.get_logger().info("turn signal: "+ str(self.turn_signal))
            if self.turn_signal == 1 and (len(results) == 0 or results == None or len(results[0].boxes) == 0):
                self.turn_signal = 0
                time.sleep(2)
                self.get_logger().info("I'm diving...")
                self.start_time = time.time()
                elapsed_time = 0.0
                self.step_time = (self.collision_height * 15) / 230 
                while elapsed_time < self.step_time:
                    self.drive_forward()
                    elapsed_time = time.time() - self.start_time
                self.dive_signal = 1
            
            if self.dive_signal == 1:
                self.dive_signal = 2 
            if self.dive_signal == 2 and (len(results) == 0 or results == None or len(results[0].boxes) == 0):
                self.searchGoal()

            # elif len(results) !=0 or results != None or len(results[0].boxes) != 0:
            #     if self.isCollision(self.collision_height) != True:
            #         self.drive_forward()
            #     else:
            #         self.avoid_obstacle(self.collision_height)
                    
            # elif self.dive_signal == 1 and (len(results) != 0 or results != None or len(results[0].boxes) != 0):
            #     self.dive_signal = 0
            #     self.searchGoal()
            # # else:
            # if len(results) == 0 or results == None or len(results[0].boxes) == 0:
            #     self.searchGoal()


    #moves the robot forward
    def drive_forward(self, speed = 0.05):
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
            # time.sleep(0.5)
            # self.drive_forward()           
            time.sleep(2)
            
        elif msg  < 125:
            self.get_logger().info("turning right "+str(msg))
            twist_msg.angular.z = -0.1
            twist_msg.linear.x = 0.0
            self.robot_state = 'right-turn'
            self.cmd_publisher.publish(twist_msg)
            # time.sleep(0.5)
            # self.drive_forward()
            time.sleep(2)

    #turn left or right to search for goal
    def searchGoal(self):
        twist_msg = Twist()
        if self.robot_state == 'right-turn':
            twist_msg.angular.z = 0.05
            twist_msg.linear.x = 0.0
            self.cmd_publisher.publish(twist_msg)
            self.get_logger().info("searching turning left")
            time.sleep(1)
        elif self.robot_state == 'left-turn':
            twist_msg.angular.z = -0.05
            twist_msg.linear.x = 0.0
            self.cmd_publisher.publish(twist_msg)
            self.get_logger().info("searching - turning right")
            time.sleep(1)
    
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
