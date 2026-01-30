# Sample end-effector path message
# Contains a list of waypoints for the end-effector to follow

std_msgs/Header header

# Array of poses representing waypoints
geometry_msgs/Pose[] waypoints

# Optional: time between waypoints (seconds)
float64[] time_stamps

# Path metadata
string path_id
string description
