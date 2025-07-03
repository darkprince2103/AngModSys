from tkinter import *
from PIL import Image, ImageTk
import os
import random

os.chdir(os.path.dirname(os.path.abspath(__file__)))


root = Tk()
root.title("Nagel-Schreckenberg Traffic Simulation")
root.configure(bg="#2e2e2e")  # Dark gray background
root.state('zoomed')  # Full size window (Windows)
# Create main frames
left_frame = Frame(root, bg="#2e2e2e")
right_frame = Frame(root, bg="#232323")

left_frame.place(relx=0, rely=0, relwidth=0.66, relheight=1)
right_frame.place(relx=0.66, rely=0, relwidth=0.34, relheight=1)

# Canvas for drawing on the left
canvas = Canvas(left_frame, bg="#2e2e2e", highlightthickness=0)
canvas.pack(fill=BOTH, expand=True)

# Load images
car_img = Image.open("img/car.png")
car_img = car_img.resize((64, 64), Image.Resampling.LANCZOS)
car_photo = ImageTk.PhotoImage(car_img)

car_slow_img = Image.open("img/car_slow.png")
car_slow_img = car_slow_img.resize((64, 64), Image.Resampling.LANCZOS)
car_slow_photo = ImageTk.PhotoImage(car_slow_img)

lkw_img = Image.open("img/lkw.png")
lkw_img = lkw_img.resize((64, 64), Image.Resampling.LANCZOS)
lkw_photo = ImageTk.PhotoImage(lkw_img)

lkw_slow_img = Image.open("img/lkw_slow.png")
lkw_slow_img = lkw_slow_img.resize((64, 64), Image.Resampling.LANCZOS)
lkw_slow_photo = ImageTk.PhotoImage(lkw_slow_img)

# Nagel-Schreckenberg Model Parameters
CELL_SIZE = 10  # Size of each cell in pixels
# Speed parameters in km/h
MIN_SPEED_CAR = 80    # Minimum speed for cars in km/h
MAX_SPEED_CAR = 200   # Maximum speed for cars in km/h
BASE_SPEED_TRUCK = 80 # Base speed for trucks in km/h
TRUCK_SPEED_VARIATION = 5  # ±5 km/h variation for trucks

# Convert km/h to simulation units (cells per step)
KMH_TO_CELLS = 0.05  # Conversion factor: 1 km/h = 0.05 cells/step (reduced for slower visual movement)

BRAKE_PROB = 0.0005  # Probability of random braking
NUM_CELLS = 150  # Number of cells on the road
MOVEMENT_FRACTION = 0.15  # Fraction of cell to move per step (reduced for slower visual movement)

# Vehicle counts
num_cars = 18
num_trucks = 7

# Vehicle class
class Vehicle:
    def __init__(self, position, vehicle_type="car", lane=0):
        self.position = position
        self.fractional_position = 0.0  # For smooth movement
        self.speed = 0
        self.vehicle_type = vehicle_type  # "car" or "truck"          # Add realistic speed variation for individual vehicles
        if vehicle_type == "car":
            # Cars have diverse speeds between min-max km/h (use current parameter values)
            min_speed = MIN_SPEED_CAR if 'car_min_var' not in globals() else car_min_var.get()
            max_speed = MAX_SPEED_CAR if 'car_max_var' not in globals() else car_max_var.get()
            self.speed_kmh = random.randint(min_speed, max_speed)
            self.max_speed = max(1, int(self.speed_kmh * KMH_TO_CELLS))
        else:
            # Trucks use the configured speed range
            if 'truck_min_var' in globals() and 'truck_max_var' in globals():
                min_speed = truck_min_var.get()
                max_speed = truck_max_var.get()
                self.speed_kmh = random.randint(min_speed, max_speed)
            else:
                # Fallback to original truck speed logic
                speed_variation = random.randint(-TRUCK_SPEED_VARIATION, TRUCK_SPEED_VARIATION)
                self.speed_kmh = BASE_SPEED_TRUCK + speed_variation
            self.max_speed = max(1, int(self.speed_kmh * KMH_TO_CELLS))
        self.length = 1 if vehicle_type == "car" else 2  # cells occupied
        self.lane = lane  # 0 = upper lane, 1 = lower lane
        self.lane_change_cooldown = 0  # Prevent frequent lane changes
        self.preferred_lane = 1  # All vehicles prefer bottom lane
        
    def get_image(self, is_slowing=False):
        if self.vehicle_type == "car":
            return car_slow_photo if is_slowing else car_photo
        else:
            return lkw_slow_photo if is_slowing else lkw_photo

# Traffic simulation
vehicles = []
is_running = False
step_count = 0

def init_traffic():
    global vehicles
    vehicles.clear()
    # Add some random vehicles in both lanes
    occupied_positions = {0: set(), 1: set()}  # Track positions for each lane
      # Use the new car and truck counts
    num_vehicles = num_cars + num_trucks
    
    # Place cars first
    for i in range(num_cars):
        attempts = 0
        while attempts < 100:  # Try up to 100 times to place a vehicle
            pos = random.randint(0, NUM_CELLS - 2)
            # All vehicles prefer bottom lane (lane 1), with 80% chance for bottom lane
            lane = 1 if random.random() < 0.8 else 0
            
            # Check if position is free (cars take 1 cell)
            if not any(p in occupied_positions[lane] for p in range(pos, pos + 1)):
                vehicle = Vehicle(pos, "car", lane)
                vehicle.speed = random.randint(0, vehicle.max_speed)
                vehicles.append(vehicle)
                occupied_positions[lane].add(pos)
                break
            attempts += 1
    
    # Place trucks
    for i in range(num_trucks):
        attempts = 0
        while attempts < 100:  # Try up to 100 times to place a vehicle
            pos = random.randint(0, NUM_CELLS - 3)
            # All vehicles prefer bottom lane (lane 1), with 80% chance for bottom lane
            lane = 1 if random.random() < 0.8 else 0
            
            # Check if position is free (trucks take 2 cells)
            if not any(p in occupied_positions[lane] for p in range(pos, pos + 2)):
                vehicle = Vehicle(pos, "truck", lane)
                vehicle.speed = random.randint(0, vehicle.max_speed)
                vehicles.append(vehicle)
                for p in range(pos, pos + 2):
                    occupied_positions[lane].add(p)
                break
            attempts += 1

def get_distance_to_next_vehicle(vehicle, lane=None):
    """Calculate distance to next vehicle ahead in specified lane"""
    if lane is None:
        lane = vehicle.lane
        
    min_distance = NUM_CELLS
    for other in vehicles:
        if other != vehicle and other.lane == lane:
            # Calculate wrapped distance considering both vehicles' positions and lengths
            vehicle_rear = vehicle.position + vehicle.length
            
            if other.position >= vehicle_rear:
                # Other vehicle is ahead in normal order
                distance = other.position - vehicle_rear
            else:
                # Other vehicle is ahead but wrapped around
                distance = (NUM_CELLS - vehicle_rear) + other.position
            
            if distance < min_distance and distance >= 0:
                min_distance = distance
    
    return max(0, min_distance)  # Ensure non-negative distance

def is_position_occupied(position, lane, length=1, exclude_vehicle=None):
    """Check if a position range is occupied by any vehicle"""
    for vehicle in vehicles:
        if vehicle == exclude_vehicle or vehicle.lane != lane:
            continue
        
        # Check if any part of the new vehicle would overlap with existing vehicle
        vehicle_start = vehicle.position
        vehicle_end = (vehicle.position + vehicle.length - 1) % NUM_CELLS
        
        for i in range(length):
            check_pos = (position + i) % NUM_CELLS
            
            # Handle wrap-around case
            if vehicle_start <= vehicle_end:
                # Normal case - no wrap around
                if vehicle_start <= check_pos <= vehicle_end:
                    return True
            else:
                # Vehicle wraps around the end of the road
                if check_pos >= vehicle_start or check_pos <= vehicle_end:
                    return True
    return False

def can_change_lane(vehicle, target_lane):
    """Check if vehicle can safely change to target lane"""
    if vehicle.lane == target_lane or vehicle.lane_change_cooldown > 0:
        return False
    
    # Check if target position would be occupied (with safety margins)
    safety_margin = 2
    for check_pos in range(vehicle.position - safety_margin, vehicle.position + vehicle.length + safety_margin):
        if is_position_occupied(check_pos % NUM_CELLS, target_lane, 1, exclude_vehicle=vehicle):
            return False
    
    # Check safety distances to other vehicles in target lane
    safe_distance_ahead = max(vehicle.speed + 3, 4)  # Safety margin ahead
    safe_distance_behind = 4  # Safety margin behind
    
    for other in vehicles:
        if other != vehicle and other.lane == target_lane:
            # Calculate distance to other vehicle
            if other.position >= vehicle.position:
                distance_to_other = other.position - vehicle.position - vehicle.length
            else:
                distance_to_other = (NUM_CELLS - vehicle.position - vehicle.length) + other.position
            
            # Calculate distance from other vehicle to us
            if vehicle.position >= other.position:
                distance_from_other = vehicle.position - other.position - other.length
            else:
                distance_from_other = (NUM_CELLS - other.position - other.length) + vehicle.position
            
            # Check if distances are safe
            if distance_to_other < safe_distance_ahead or distance_from_other < safe_distance_behind:
                return False
    
    return True

def get_lane_change_benefit(vehicle):
    """Determine if lane change would be beneficial"""
    current_distance = get_distance_to_next_vehicle(vehicle)
    other_lane = 1 - vehicle.lane
    other_distance = get_distance_to_next_vehicle(vehicle, other_lane)
    
    # Check if vehicle is being slowed down significantly
    is_significantly_slowed = vehicle.speed < (vehicle.max_speed * 0.7)  # Less than 70% of max speed
    
    if vehicle.lane == 0 and other_lane == 1:  # Moving to preferred lane (bottom)
        # Always want to return to bottom lane if possible
        return other_distance > current_distance + 2  # More lenient for returning to preferred lane
    elif vehicle.lane == 1 and other_lane == 0:  # Moving away from preferred lane (overtaking)
        # Only move to upper lane if significantly slowed AND much more space available
        return is_significantly_slowed and other_distance > current_distance + 8  # Strict for leaving preferred lane
    
    # Default: only change if significantly slowed and other lane is much better
    return is_significantly_slowed and other_distance > current_distance + 5

def update_simulation():
    """One step of Nagel-Schreckenberg model with lane changing"""
    global step_count
    step_count += 1
    
    # Reduce lane change cooldown for all vehicles
    for vehicle in vehicles:
        if vehicle.lane_change_cooldown > 0:
            vehicle.lane_change_cooldown -= 1      # Step 0: Lane changing with preference to return to bottom lane
    for vehicle in vehicles:
        # Check if lane change would be beneficial and safe
        other_lane = 1 - vehicle.lane
        
        # Determine lane change probability based on current lane and situation
        if vehicle.lane == 0 and other_lane == 1:  # Returning to preferred bottom lane
            lane_change_probability = 0.3  # 30% chance to return to bottom lane when safe
            # Even higher if not being slowed down (finished overtaking)
            if vehicle.speed >= (vehicle.max_speed * 0.8):  # Going at 80%+ of max speed
                lane_change_probability = 0.5  # 50% chance when moving well
        elif vehicle.lane == 1 and other_lane == 0:  # Moving to upper lane (overtaking)
            lane_change_probability = 0.005  # 0.5% random chance
            # Higher chance if vehicle is being slowed down significantly
            if vehicle.speed < (vehicle.max_speed * 0.7):  # Less than 70% of max speed
                lane_change_probability = 0.15  # 15% chance when slowed down
        else:
            lane_change_probability = 0.005  # Default low probability
        
        if (can_change_lane(vehicle, other_lane) and 
            get_lane_change_benefit(vehicle) and 
            random.random() < lane_change_probability):
            vehicle.lane = other_lane
            vehicle.lane_change_cooldown = 8  # Longer cooldown to prevent frequent switching
    
    # Step 1: Acceleration
    for vehicle in vehicles:
        if vehicle.speed < vehicle.max_speed:
            vehicle.speed += 1
    
    # Step 2: Braking (collision avoidance)
    for vehicle in vehicles:
        distance = get_distance_to_next_vehicle(vehicle)
        if vehicle.speed > distance:
            vehicle.speed = max(0, distance)
            
        # Lower lane vehicles can't go faster than upper lane traffic ahead
        if vehicle.lane == 1:
            upper_distance = get_distance_to_next_vehicle(vehicle, lane=0)
            if vehicle.speed > upper_distance:
                vehicle.speed = max(0, upper_distance)      # Step 3: Random braking
    for vehicle in vehicles:
        current_brake_prob = brake_var.get() if 'brake_var' in globals() else BRAKE_PROB
        if vehicle.speed > 0 and random.random() < current_brake_prob:
            vehicle.speed -= 1# Step 4: Movement with proper collision prevention
    for vehicle in vehicles:
        if vehicle.speed > 0:
            # Calculate fractional movement
            movement = vehicle.speed * MOVEMENT_FRACTION
            vehicle.fractional_position += movement
            
            # Check if we need to move to next cell
            if vehicle.fractional_position >= 1.0:
                cells_to_move = int(vehicle.fractional_position)
                vehicle.fractional_position -= cells_to_move
                
                new_position = (vehicle.position + cells_to_move) % NUM_CELLS
                
                # Simple collision check - only check final destination with small buffer
                collision_buffer = 1
                collision_detected = False
                
                for i in range(vehicle.length + collision_buffer):
                    check_pos = (new_position + i) % NUM_CELLS
                    if is_position_occupied(check_pos, vehicle.lane, 1, exclude_vehicle=vehicle):
                        collision_detected = True
                        break
                
                if not collision_detected:
                    vehicle.position = new_position
                else:
                    # If collision would occur, reduce speed and reset fractional position
                    vehicle.speed = max(0, vehicle.speed - 1)
                    vehicle.fractional_position = 0

# Animation variables
is_animating = False

def redraw(event=None):
    canvas.delete("all")
    w = canvas.winfo_width()
    h = canvas.winfo_height()
    line_y = h // 2
    
    # Draw thick white lines (outer lines solid, middle line dashed)
    canvas.create_line(50, line_y-120, w-50, line_y-120, fill="white", width=12)  # Upper line (solid)
    canvas.create_line(50, line_y+120, w-50, line_y+120, fill="white", width=12)  # Lower line (solid)
    
    # Draw dashed middle line
    dash_length = 30
    gap_length = 20
    x = 50
    while x < w - 50:
        end_x = min(x + dash_length, w - 50)
        canvas.create_line(x, line_y, end_x, line_y, fill="white", width=12)
        x += dash_length + gap_length
      # Draw vehicles
    road_width = w - 100  # Available road width
    cell_pixel_width = road_width / NUM_CELLS
    
    for vehicle in vehicles:
        # Calculate pixel position with smooth fractional movement
        base_pixel_x = 50 + (vehicle.position * cell_pixel_width)
        fractional_offset = vehicle.fractional_position * cell_pixel_width
        pixel_x = base_pixel_x + fractional_offset + (cell_pixel_width / 2)
        
        # Different y positions for different lanes
        if vehicle.lane == 0:  # Upper lane
            pixel_y = line_y - 60
        else:  # Lower lane
            pixel_y = line_y + 60
        
        # Determine if vehicle is slowing (speed < max_speed/2)
        is_slowing = vehicle.speed < vehicle.max_speed / 2
          # Get appropriate image
        img = vehicle.get_image(is_slowing)        
        # Draw vehicle
        canvas.create_image(pixel_x, pixel_y, image=img)

def start_simulation():
    global is_running
    if not is_running:
        init_traffic()  # Only create new traffic when starting fresh
        is_running = True
        update_parameter_controls()  # Disable controls
        start_button.config(state="disabled")  # Disable start button when running
        stop_button.config(text="Pause Simulation")  # Change text to pause
        run_simulation()

def stop_simulation():
    global is_running
    if is_running:
        # Pause the simulation (keep existing vehicles)
        is_running = False
        update_parameter_controls()  # Enable controls
        stop_button.config(text="Resume Simulation")  # Change text to resume
    else:
        # Resume the simulation (don't reinitialize traffic)
        is_running = True
        update_parameter_controls()  # Disable controls
        stop_button.config(text="Pause Simulation")  # Change text to pause
        run_simulation()  # Continue with existing vehicles

def run_simulation():
    if is_running:
        update_simulation()
        redraw()
        draw_max_speed_diagram()  # Update diagram every simulation step
        root.after(50, run_simulation)  # Update every 50ms for very smooth animation

def reset_simulation():
    global is_running
    is_running = False
    update_parameter_controls()  # Enable controls
    start_button.config(state="normal")  # Enable start button
    stop_button.config(text="Pause Simulation")  # Reset button text
    init_traffic()
    redraw()
    draw_max_speed_diagram()  # Update diagram after reset

# Add buttons to right frame
start_button = Button(right_frame, text="Start Simulation", command=start_simulation, 
                     bg="#4a8a4a", fg="white", font=("Arial", 12), 
                     relief="raised", bd=2, padx=20, pady=10)
start_button.pack(pady=10)

stop_button = Button(right_frame, text="Pause Simulation", command=stop_simulation, 
                    bg="#8a4a4a", fg="white", font=("Arial", 12), 
                    relief="raised", bd=2, padx=20, pady=10)
stop_button.pack(pady=10)

reset_button = Button(right_frame, text="Reset", command=reset_simulation, 
                     bg="#4a4a8a", fg="white", font=("Arial", 12), 
                     relief="raised", bd=2, padx=20, pady=10)
reset_button.pack(pady=10)

# Add parameter controls
params_label = Label(right_frame, text="Simulation Parameters", bg="#232323", fg="white", 
      font=("Arial", 14, "bold"))
params_label.pack(pady=(30, 10))

# Create parameter frame
param_frame = Frame(right_frame, bg="#232323")
param_frame.pack(pady=10, fill=X, padx=20)

# Brake Probability control
brake_frame = Frame(param_frame, bg="#232323")
brake_frame.pack(fill=X, pady=5)
Label(brake_frame, text="Brake Probability:", bg="#232323", fg="white", 
      font=("Arial", 10)).pack(side=LEFT)
brake_var = DoubleVar(value=BRAKE_PROB)
brake_scale = Scale(brake_frame, from_=0.0001, to=0.05, resolution=0.0001, 
                   orient=HORIZONTAL, variable=brake_var, bg="#232323", fg="white",
                   highlightthickness=0, length=150)
brake_scale.pack(side=RIGHT)

# Vehicle count controls
count_frame = Frame(param_frame, bg="#232323")
count_frame.pack(fill=X, pady=5)
Label(count_frame, text="Vehicle Counts:", bg="#232323", fg="white", 
      font=("Arial", 10)).pack(side=LEFT)
count_subframe = Frame(count_frame, bg="#232323")
count_subframe.pack(side=RIGHT)

# Car count
car_count_var = IntVar(value=18)
Label(count_subframe, text="Cars:", bg="#232323", fg="white", font=("Arial", 9)).pack(side=LEFT)
car_count_scale = Scale(count_subframe, from_=0, to=30, orient=HORIZONTAL, 
                       variable=car_count_var, bg="#232323", fg="white",
                       highlightthickness=0, length=60)
car_count_scale.pack(side=LEFT, padx=(0,10))

# Truck count
truck_count_var = IntVar(value=7)
Label(count_subframe, text="Trucks:", bg="#232323", fg="white", font=("Arial", 9)).pack(side=LEFT)
truck_count_scale = Scale(count_subframe, from_=0, to=10, orient=HORIZONTAL, 
                         variable=truck_count_var, bg="#232323", fg="white",
                         highlightthickness=0, length=60)
truck_count_scale.pack(side=LEFT)

# Speed range for cars
car_speed_frame = Frame(param_frame, bg="#232323")
car_speed_frame.pack(fill=X, pady=5)
Label(car_speed_frame, text="Car Speed Range (km/h):", bg="#232323", fg="white", 
      font=("Arial", 10)).pack(side=LEFT)
car_min_var = IntVar(value=MIN_SPEED_CAR)
car_max_var = IntVar(value=MAX_SPEED_CAR)
car_speed_subframe = Frame(car_speed_frame, bg="#232323")
car_speed_subframe.pack(side=RIGHT)
car_min_scale = Scale(car_speed_subframe, from_=60, to=120, orient=HORIZONTAL, 
                     variable=car_min_var, bg="#232323", fg="white",
                     highlightthickness=0, length=70)
car_min_scale.pack(side=LEFT)
Label(car_speed_subframe, text="-", bg="#232323", fg="white").pack(side=LEFT)
car_max_scale = Scale(car_speed_subframe, from_=150, to=250, orient=HORIZONTAL, 
                     variable=car_max_var, bg="#232323", fg="white",
                     highlightthickness=0, length=70)
car_max_scale.pack(side=LEFT)

# Speed range for trucks
truck_speed_frame = Frame(param_frame, bg="#232323")
truck_speed_frame.pack(fill=X, pady=5)
Label(truck_speed_frame, text="Truck Speed Range (km/h):", bg="#232323", fg="white", 
      font=("Arial", 10)).pack(side=LEFT)
truck_min_var = IntVar(value=BASE_SPEED_TRUCK - TRUCK_SPEED_VARIATION)
truck_max_var = IntVar(value=BASE_SPEED_TRUCK + TRUCK_SPEED_VARIATION)
truck_speed_subframe = Frame(truck_speed_frame, bg="#232323")
truck_speed_subframe.pack(side=RIGHT)
truck_min_scale = Scale(truck_speed_subframe, from_=60, to=90, orient=HORIZONTAL, 
                       variable=truck_min_var, bg="#232323", fg="white",
                       highlightthickness=0, length=70)
truck_min_scale.pack(side=LEFT)
Label(truck_speed_subframe, text="-", bg="#232323", fg="white").pack(side=LEFT)
truck_max_scale = Scale(truck_speed_subframe, from_=80, to=120, orient=HORIZONTAL, 
                       variable=truck_max_var, bg="#232323", fg="white",
                       highlightthickness=0, length=70)
truck_max_scale.pack(side=LEFT)

# Apply changes button
apply_button = Button(param_frame, text="Apply Changes", 
                     command=lambda: apply_parameter_changes(),
                     bg="#4a6a4a", fg="white", font=("Arial", 10), 
                     relief="raised", bd=2, padx=10, pady=5)
apply_button.pack(pady=10)

def apply_parameter_changes():
    """Apply parameter changes when simulation is stopped"""
    global BRAKE_PROB, MIN_SPEED_CAR, MAX_SPEED_CAR, BASE_SPEED_TRUCK, TRUCK_SPEED_VARIATION
    global num_cars, num_trucks
    if not is_running:
        BRAKE_PROB = brake_var.get()
        MIN_SPEED_CAR = car_min_var.get()
        MAX_SPEED_CAR = car_max_var.get()
        
        # Update truck speed range
        truck_min = truck_min_var.get()
        truck_max = truck_max_var.get()
        BASE_SPEED_TRUCK = (truck_min + truck_max) // 2
        TRUCK_SPEED_VARIATION = (truck_max - truck_min) // 2
        
        # Update vehicle counts
        num_cars = car_count_var.get()
        num_trucks = truck_count_var.get()
        
        # Reinitialize traffic with new parameters
        init_traffic()
        redraw()
        draw_max_speed_diagram()

def update_parameter_controls():
    """Enable/disable parameter controls based on simulation state"""
    state = "disabled" if is_running else "normal"
    brake_scale.config(state=state)
    
    # Update count controls
    car_count_scale.config(state=state)
    truck_count_scale.config(state=state)
    
    # Update speed range controls
    car_min_scale.config(state=state)
    car_max_scale.config(state=state)
    truck_min_scale.config(state=state)
    truck_max_scale.config(state=state)
    
    apply_button.config(state=state)

# Add statistics display
stats_frame = Frame(right_frame, bg="#232323")
stats_frame.pack(pady=(30, 10), fill=X, padx=20)

avg_speed_label = Label(stats_frame, text="Avg Speed: 0 km/h", bg="#232323", fg="white", 
                       font=("Arial", 10))
avg_speed_label.pack()

# Add diagram for max speed statistics
diagram_frame = Frame(right_frame, bg="#232323")
diagram_frame.pack(pady=(20, 10), fill=BOTH, expand=True, padx=20)

Label(diagram_frame, text="Vehicle Speed Comparison", bg="#232323", fg="white", 
      font=("Arial", 12, "bold")).pack(pady=(0, 10))

# Canvas for the diagram
diagram_canvas = Canvas(diagram_frame, bg="#1a1a1a", height=200, highlightthickness=1, 
                       highlightbackground="#444444")
diagram_canvas.pack(fill=BOTH, expand=True)

def calculate_average_speed():
    """Calculate average speed of all vehicles in km/h"""
    if not vehicles:
        return 0, 0, 0
    
    # Convert current simulation speeds back to km/h using the same conversion as display
    car_speeds = []
    truck_speeds = []
    all_speeds = []
    
    for v in vehicles:
        # Convert simulation speed back to km/h
        current_speed_kmh = v.speed / KMH_TO_CELLS
        
        all_speeds.append(current_speed_kmh)
        if v.vehicle_type == "car":
            car_speeds.append(current_speed_kmh)
        else:
            truck_speeds.append(current_speed_kmh)
    
    avg_car = sum(car_speeds) / len(car_speeds) if car_speeds else 0
    avg_truck = sum(truck_speeds) / len(truck_speeds) if truck_speeds else 0
    avg_all = sum(all_speeds) / len(all_speeds) if all_speeds else 0
    
    return avg_all, avg_car, avg_truck

def calculate_max_speed_stats():
    """Calculate how many vehicles are driving at max speed"""
    if not vehicles:
        return 0, 0, 0, 0
    
    cars_at_max = sum(1 for v in vehicles if v.vehicle_type == "car" and v.speed >= v.max_speed)
    trucks_at_max = sum(1 for v in vehicles if v.vehicle_type == "truck" and v.speed >= v.max_speed)
    total_cars = sum(1 for v in vehicles if v.vehicle_type == "car")
    total_trucks = sum(1 for v in vehicles if v.vehicle_type == "truck")
    
    return cars_at_max, trucks_at_max, total_cars, total_trucks

def draw_max_speed_diagram():
    """Draw diagram showing current speed vs max speed for each vehicle"""
    diagram_canvas.delete("all")
    
    if not vehicles:
        return
    
    w = diagram_canvas.winfo_width()
    h = diagram_canvas.winfo_height()
    
    if w <= 1 or h <= 1:  # Canvas not ready yet
        return
    
    # Sort vehicles by type for better visualization
    cars = [v for v in vehicles if v.vehicle_type == "car"]
    trucks = [v for v in vehicles if v.vehicle_type == "truck"]
    
    # Calculate layout
    margin = 20
    available_width = w - 2 * margin
    bar_spacing = 2
    total_vehicles = len(vehicles)
    
    if total_vehicles == 0:
        return
        
    bar_width = max(3, (available_width - (total_vehicles - 1) * bar_spacing) // total_vehicles)
    max_bar_height = h - 60
    
    # Draw title
    diagram_canvas.create_text(w//2, 15, text="Current Speed vs Max Speed", 
                              fill="white", font=("Arial", 9, "bold"))
    
    # Draw legend
    legend_y = 30
    diagram_canvas.create_rectangle(margin, legend_y, margin + 10, legend_y + 10, 
                                   fill="#4a8a4a", outline="#6aba6a")
    diagram_canvas.create_text(margin + 15, legend_y + 5, text="Cars", 
                              fill="white", font=("Arial", 7), anchor="w")
    
    diagram_canvas.create_rectangle(margin + 60, legend_y, margin + 70, legend_y + 10, 
                                   fill="#8a4a4a", outline="#ba6a6a")
    diagram_canvas.create_text(margin + 75, legend_y + 5, text="Trucks", 
                              fill="white", font=("Arial", 7), anchor="w")
    
    # Draw vehicles
    x_pos = margin
    
    # Draw all vehicles in order (cars first, then trucks for consistency)
    all_vehicles_sorted = cars + trucks
    
    for vehicle in all_vehicles_sorted:
        if x_pos + bar_width > w - margin:
            break
            
        # Calculate current speed percentage based on simulation values
        speed_percentage = vehicle.speed / vehicle.max_speed if vehicle.max_speed > 0 else 0
        
        # Choose colors based on vehicle type
        if vehicle.vehicle_type == "car":
            bg_color = "#2a4a2a"
            fg_color = "#4a8a4a"
            text_color = "#6aba6a"
        else:  # truck
            bg_color = "#4a2a2a"
            fg_color = "#8a4a4a" 
            text_color = "#ba6a6a"
        
        # Max speed bar (background)
        max_bar_height_actual = int(0.7 * max_bar_height)
        bar_y_start = h - 25
        bar_y_max = bar_y_start - max_bar_height_actual
        
        # Background bar (max speed potential)
        diagram_canvas.create_rectangle(x_pos, bar_y_max, x_pos + bar_width, bar_y_start, 
                                       fill=bg_color, outline="white", width=1)
        
        # Current speed bar (foreground) - this will change as vehicle speeds change
        current_bar_height = max(1, int(speed_percentage * max_bar_height_actual))
        bar_y_current = bar_y_start - current_bar_height
          # Only draw current speed bar if there's actually some speed
        if current_bar_height > 0:
            diagram_canvas.create_rectangle(x_pos, bar_y_current, x_pos + bar_width, bar_y_start, 
                                           fill=fg_color, outline="white", width=1)
        
        x_pos += bar_width + bar_spacing
    
    # Draw scale labels
    diagram_canvas.create_text(margin - 5, h - 25, text="0%", 
                              fill="white", font=("Arial", 6), anchor="e")
    diagram_canvas.create_text(margin - 5, h - 25 - int(0.7 * max_bar_height), text="100%", 
                              fill="white", font=("Arial", 6), anchor="e")    # Draw summary statistics with real-time data
    if vehicles:
        total_at_max = sum(1 for v in vehicles if v.speed >= v.max_speed)
        avg_speed_pct = sum(v.speed / v.max_speed for v in vehicles if v.max_speed > 0) / len(vehicles) * 100
        
        summary_text = f"At Max: {total_at_max}/{len(vehicles)} | Avg: {avg_speed_pct:.0f}%"
        diagram_canvas.create_text(w - margin, h - 10, text=summary_text, 
                                  fill="white", font=("Arial", 7), anchor="e")

def update_stats():
    # Only update average speed when simulation is running
    if is_running:
        avg_all, avg_car, avg_truck = calculate_average_speed()
        speed_text = f"Avg Speed: {avg_all:.1f} km/h (Cars: {avg_car:.1f}, Trucks: {avg_truck:.1f})"
        avg_speed_label.config(text=speed_text)
    else:
        avg_speed_label.config(text="Avg Speed: -- km/h (Simulation stopped)")
    
    # Force update the diagram every time
    root.after_idle(draw_max_speed_diagram)
    
    # Continue updating stats regardless of simulation state
    root.after(100, update_stats)

# Start stats update
root.after(100, update_stats)

canvas.bind("<Configure>", redraw)
diagram_canvas.bind("<Configure>", lambda e: draw_max_speed_diagram())

# Initialize the simulation and parameter controls
init_traffic()
update_parameter_controls()  # Set initial control states
# Call redraw initially to show content immediately
root.after(1, redraw)  # Use after to ensure canvas is fully initialized
root.mainloop()