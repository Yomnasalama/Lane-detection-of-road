# Lane-detection-of-road
#To run Pipline video:
  sh script1.sh input_path output_path '--piplinemood'
#To run debug video:
  sh.script1.sh input_path output_path '--debugmood'

#Detection Pipeline
  1. Get edges positions and directions using Sobel
  2. Applying perspective warping
  3. Detect using sliding windows
  4. Draw lanes
  5. get_curvature of road
  
