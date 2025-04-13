# # import cv2
# # from ultralytics import YOLO
# # from rclpy.node import Node
# # from sensor_msgs.msg import Image
# # from cv_bridge import CvBridge


# # class Object_Detection(Node):
# #     def __init__(self):
# #         super().__init__('object_detection_node')
# #         # Load YOLO model
# #         self.model = YOLO("yolov8n.pt")
        
# #         self.subscription = self.create_subscription(
# #             Image,
# #             '/color/image',
# #             self.bounding_box_callback,
# #             10
# #         )
        
# #     def bounding_box_callback(self, stream):
# #         stream = cv2.CvBridge().imgmsg_to_cv2(stream, desired_encoding='bgr8')
# #         cap = cv2.VideoCapture(stream)
# #         while cap.isOpened():
# #             ret, frame = cap.read()
# #             if not ret:
# #                 break

# #             # Run predictions on the frame
# #             results = self.model(frame)
            
# #             # Plot the predictions (bounding boxes, labels, etc.)
# #             annotated_frame = results[0].plot()  # Get the frame with plotted boxes
            
# #             # Iterate through each bounding box
# #             for result in results[0].boxes:
# #                 # Extract coordinates for the bounding box in xywh format
# #                 xywh = result.xywh[0].cpu().numpy()  # [x_center, y_center, width, height]
# #                 x_center, y_center = int(xywh[0]), int(xywh[1])  # Center coordinates in pixels
                
# #                 # Mark the center point on the frame
# #                 cv2.circle(annotated_frame, (x_center, y_center), radius=5, color=(0, 0, 255), thickness=-1)
                
# #                 # Optionally, add text at the center (e.g., 'x')
# #                 # cv2.putText(annotated_frame, 'x', (x_center, y_center), 
# #                 #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# #             # Display the annotated frame
# #             cv2.imshow("YOLOv8 Prediction", annotated_frame)

# #             # Break on 'q' key
# #             if cv2.waitKey(1) & 0xFF == ord('q'):
# #                 break
# #         cap.release()
# #         cv2.destroyAllWindows()

# # def main(args=None):
# #     rclpy.init(args=args)
# #     object_detector = Object_Detection()
# #     rclpy.spin(object_detector)
# #     object_detector.destroy_node()
# #     rclpy.shutdown()

# # if __name__ == '__main__':
# #     main()


# ############################################################################
# #OBJECT DETECTION  FIRST TRY 
# # import time
# # import rclpy
# # from rclpy.node import Node
# # from sensor_msgs.msg import Image
# # import ros2_numpy as rnp
# # from ultralytics import YOLO
# # import cv2

# # class DetectionNode(Node):
# #     def __init__(self):
# #         super().__init__("ultralytics")
# #         self.detection_model = YOLO("/home/grandson/ros2_ws/src/object_detection/object_detection/best.pt")
# #         self.segmentation_model = YOLO("yolo11n-seg.pt")
        
# #         # Publisher setup
# #         self.det_image_pub = self.create_publisher(Image, "/ultralytics/detection/image", 10)
# #         # self.seg_image_pub = self.create_publisher(Image, "/ultralytics/segmentation/image", 10)
        
# #         # Subscriber setup
# #         self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.callback, 10)
        
# #         # Sleep for a second to ensure everything is initialized
# #         time.sleep(1)

# #     def callback(self, data):
# #         """Callback function to process image and publish annotated images."""
# #         array = rnp.numpify(data)

# #         # Detection
# #         if self.det_image_pub.get_subscription_count():
# #             det_result = self.detection_model(array, conf=0.90)
# #             det_annotated = det_result[0].plot(show=False)
# #             self.det_image_pub.publish(rnp.msgify(Image, det_annotated, encoding="rgb8"))

# #         # # Segmentation
# #         # if self.seg_image_pub.get_subscription_count():
# #         #     seg_result = self.segmentation_model(array)
# #         #     seg_annotated = seg_result[0].plot(show=False)
# #         #     self.seg_image_pub.publish(rnp.msgify(Image, seg_annotated, encoding="rgb8"))

# # def main(args=None):
# #     rclpy.init(args=args)
# #     node = DetectionNode()
    
# #     try:
# #         rclpy.spin(node)
# #     except KeyboardInterrupt:
# #         pass
# #     finally:
# #         node.destroy_node()
# #         rclpy.shutdown()

# # if __name__ == "__main__":
# #     main()



# ##########################################################################

# #OBJECT DETECTION WITH STRING MSG
# # import time
# # import rclpy
# # from rclpy.node import Node
# # from sensor_msgs.msg import Image
# # from std_msgs.msg import String
# # import ros2_numpy as rnp
# # from ultralytics import YOLO

# # class DetectionNode(Node):
# #     def __init__(self):
# #         super().__init__("ultralytics")
# #         self.detection_model = YOLO("yolo11n.pt")
        
# #         # Publisher for detected classes
# #         self.classes_pub = self.create_publisher(String, "/ultralytics/detection/classes", 10)
        
# #         # Subscriber for the camera image
# #         self.create_subscription(Image, "/color/image", self.callback, 10)
        
# #         # Sleep for a second to ensure everything is initialized
# #         time.sleep(1)

# #     def callback(self, data):
# #         """Callback function to process image and publish detected class names."""
# #         array = rnp.numpify(data)

# #         if self.classes_pub.get_subscription_count():
# #             det_result = self.detection_model(array)
# #             classes = det_result[0].boxes.cls.cpu().numpy().astype(int)
# #             names = [det_result[0].names[i] for i in classes]
# #             class_names = ', '.join(names)
# #             self.classes_pub.publish(String(data=class_names))

# # def main(args=None):
# #     rclpy.init(args=args)
# #     node = DetectionNode()
    
# #     try:
# #         rclpy.spin(node)
# #     except KeyboardInterrupt:
# #         pass
# #     finally:
# #         node.destroy_node()
# #         rclpy.shutdown()

# # if __name__ == "__main__":
# #     main()


# ###################################################################################
# #OBJECT DETECTION WITH DEPTH
# # import time
# # import numpy as np
# # import rclpy
# # from rclpy.node import Node
# # from sensor_msgs.msg import Image
# # from std_msgs.msg import String
# # from ultralytics import YOLO
# # import cv2
# # import ros2_numpy as rnp

# # class DetectionNode(Node):
# #     def __init__(self):
# #         super().__init__("ultralytics_detection")
        
# #         # Load the object detection model
# #         self.detection_model = YOLO("yolo11n.pt")
        
# #         # Publisher for detected object distances
# #         self.classes_pub = self.create_publisher(String, "/ultralytics/detection/distance", 10)
        
# #         # Publisher for annotated image
# #         self.image_pub = self.create_publisher(Image, "/ultralytics/detection/image", 10)
        
        
# #         # Create a variable to store the latest RGB image
# #         self.rgb_image = None
        
# #         # Subscribe to the RGB image topic
# #         self.create_subscription(Image, "/color/preview/image", self.rgb_callback, 10)
        
# #         # Subscribe to the depth image topic
# #         self.create_subscription(Image, "/stereo/depth/compressedDepth", self.depth_callback, 10)
        
# #         # Sleep for a second to ensure everything is initialized
# #         time.sleep(1)

# #     def rgb_callback(self, msg):
# #         """Callback to store the latest RGB image."""
# #         self.rgb_image = rnp.numpify(msg)

# #     def depth_callback(self, data):
# #         """Process depth image and detect objects in the RGB image."""
# #         if self.rgb_image is None:
# #             self.get_logger().warn("Waiting for RGB image...")
# #             return
        
# #         # Convert ROS depth image to numpy array
# #         depth_image = rnp.numpify(data)
        
# #         # Run object detection model on the RGB image
# #         result = self.detection_model(self.rgb_image)
# #         all_objects = []
# #         annotated_image = self.rgb_image.copy()

# #         # Process detected objects
# #         for box in result[0].boxes:
# #             x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
# #             class_index = int(box.cls.cpu().numpy())
# #             name = result[0].names[class_index]
            
# #             # Extract depth values from the bounding box region
# #             obj_depth = depth_image[y1:y2, x1:x2]
# #             obj_depth = obj_depth[~np.isnan(obj_depth)]  # Remove NaN values

# #             # Calculate the average distance if objects are detected, otherwise set to infinity
# #             avg_distance = np.mean(obj_depth) if obj_depth.size > 0 else np.inf
# #             all_objects.append(f"{name}: {avg_distance:.2f} meters")
            
# #             # Draw bounding box and label on annotated image
# #             cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #             label = f"{name}: {avg_distance:.2f}m"
# #             cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
# #         # Publish detected object distances
# #         self.classes_pub.publish(String(data=", ".join(all_objects)))
        
# #         # Convert annotated image to ROS Image message and publish
# #         annotated_msg = rnp.msgify(Image, annotated_image, encoding='rgb8')
# #         self.image_pub.publish(annotated_msg)

# # def main(args=None):
# #     rclpy.init(args=args)
# #     node = DetectionNode()
    
# #     try:
# #         rclpy.spin(node)
# #     except KeyboardInterrupt:
# #         pass
# #     finally:
# #         node.destroy_node()
# #         rclpy.shutdown()

# # if __name__ == "__main__":
# #     main()


# ##################################################################################
# #OBJECT SEGMENTATION WITH DEPTH
# # import time
# # import numpy as np
# # import ros2_numpy as rnp
# # import rclpy
# # from rclpy.node import Node
# # from sensor_msgs.msg import Image
# # from std_msgs.msg import String
# # from ultralytics import YOLO

# # class SegmentationNode(Node):
# #     def __init__(self):
# #         super().__init__("ultralytics")
        
# #         # Load the segmentation model
# #         self.segmentation_model = YOLO("yolo11n-seg.pt")
        
# #         # Publisher for the distance of detected objects
# #         self.classes_pub = self.create_publisher(String, "/ultralytics/detection/distance", 10)
        
# #         # Create a variable to store the latest RGB image
# #         self.rgb_image = None
# #         # Subscribe to the RGB image topic
# #         self.create_subscription(Image, "color/image", self.rgb_callback, 10)

# #         # Subscriber for the depth image
# #         self.create_subscription(Image, "/stereo/depth", self.callback, 10)
        
# #         # Sleep for a second to ensure everything is initialized
# #         time.sleep(1)

# #     def rgb_callback(self, msg):
# #         """Callback to store the latest RGB image."""
# #         self.rgb_image = rnp.numpify(msg)

# #     def callback(self, data):
# #         """Callback function to process depth image and RGB image."""
# #         if self.rgb_image is None:
# #             self.get_logger().warn("Waiting for RGB image...")
# #             return
        
# #         # Convert ROS depth image to numpy array
# #         depth_image = rnp.numpify(data)

# #         # Run segmentation model on the RGB image
# #         result = self.segmentation_model(self.rgb_image)

# #         all_objects = []

# #         # Loop over detected objects in the segmentation result
# #         for index, cls in enumerate(result[0].boxes.cls):
# #             class_index = int(cls.cpu().numpy())
# #             name = result[0].names[class_index]
# #             mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)
            
# #             # Extract the depth values corresponding to the mask
# #             obj_depth = depth_image[mask == 1]
# #             obj_depth = obj_depth[~np.isnan(obj_depth)]  # Remove NaN values

# #             # Calculate the average distance if objects are detected, otherwise set to infinity
# #             avg_distance = np.mean(obj_depth) if len(obj_depth) else np.inf
# #             all_objects.append(f"{name}: {avg_distance:.2f} meters")

# #         # Publish the object distance information
# #         self.classes_pub.publish(String(data=", ".join(all_objects)))

# # def main(args=None):
# #     rclpy.init(args=args)
# #     node = SegmentationNode()
    
# #     try:
# #         rclpy.spin(node)
# #     except KeyboardInterrupt:
# #         pass
# #     finally:
# #         node.destroy_node()
# #         rclpy.shutdown()

# # if __name__ == "__main__":
# #     main()

# #####################################################################
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