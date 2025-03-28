import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
from mpl_toolkits.mplot3d import Axes3D

# Constants
NUM_ROBOTS = 4
NUM_TARGETS = 5
INITIAL_OBSTACLES = np.array([[3, 3], [7, 7], [5, 5], [2, 6], [8, 8], [4, 2], [1, 4], [6, 8]])
STEP_SIZE = 0.1
AVOIDANCE_DISTANCE = 0.6
TARGET_REACHED_THRESHOLD = 0.5
FORMATION_KEEPING_WEIGHT = 0.4
MAX_SPEED = 0.15
VELOCITY_DECAY = 0.85
COMMUNICATION_RADIUS = 2.5
FORMATION_RECOVERY_THRESHOLD = 2.0
OBSTACLE_REPULSION_STRENGTH = 2.5
LEADERSHIP_CHANGE_THRESHOLD = 1.2
LEADERSHIP_SCORE_MEMORY = 5
TARGET_DETECTION_RADIUS = 0.8
EXPLORATION_WEIGHT = 0.6
FORMATION_RECOVERY_DELAY = 15
SENSOR_LINE_SMOOTHING = 0.7
OBSTACLE_SPEED = 0.05
OBSTACLE_DIRECTION_CHANGE_PROB = 0.02
OBSTACLE_BOUNDARY_BUFFER = 0.5
MAX_SIMULATION_STEPS = 5000
INITIAL_POSITION = np.array([1, 1])

class DynamicObstacle:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1)], dtype=float)
        if np.linalg.norm(self.velocity) > 0:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * OBSTACLE_SPEED

    def update(self, obstacles):
        if random.random() < OBSTACLE_DIRECTION_CHANGE_PROB:
            new_direction = np.array([random.uniform(-1, 1), random.uniform(-1, 1)], dtype=float)
            if np.linalg.norm(new_direction) > 0:
                self.velocity = new_direction / np.linalg.norm(new_direction) * OBSTACLE_SPEED

        next_position = self.position + self.velocity

        if next_position[0] < OBSTACLE_BOUNDARY_BUFFER or next_position[0] > 10 - OBSTACLE_BOUNDARY_BUFFER:
            self.velocity[0] *= -1
        if next_position[1] < OBSTACLE_BOUNDARY_BUFFER or next_position[1] > 10 - OBSTACLE_BOUNDARY_BUFFER:
            self.velocity[1] *= -1

        for obs in obstacles:
            if obs is not self:
                distance = np.linalg.norm(next_position - obs.position)
                if distance < AVOIDANCE_DISTANCE * 2:
                    repulsion = self.position - obs.position
                    if np.linalg.norm(repulsion) > 0:
                        self.velocity += repulsion / np.linalg.norm(repulsion) * 0.01

        if np.linalg.norm(self.velocity) > 0:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * OBSTACLE_SPEED

        self.position += self.velocity

class Robot:
    def __init__(self, position, formation_offset, id):
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        self.formation_offset = formation_offset
        self.discovered_targets = set()
        self.known_targets = set()
        self.formation_deviation = 0.0
        self.avoidance_mode = False
        self.id = id
        self.leadership_score = 0.0
        self.score_history = [0.0] * LEADERSHIP_SCORE_MEMORY
        self.is_leader = False
        self.visited_positions = set()
        self.exploration_target = None
        self.exploration_time = 0
        self.sensor_line_points = []
        self.formation_recovery_timer = 0
        self.initial_position = position

    def calculate_leadership_score(self, obstacles):
        score = 0.0
        min_obstacle_dist = min(np.linalg.norm(self.position - obs.position) for obs in obstacles)
        score += np.clip(min_obstacle_dist / 5.0, 0, 1) * 2.0
        score += min(1.0, len(self.visited_positions) / 100.0) * 2.0
        score += (1.0 - min(1.0, np.linalg.norm(self.velocity) / MAX_SPEED)) * 1.0
        center_distance = np.linalg.norm(self.position - np.array([5.0, 5.0]))
        score += (1.0 - np.clip(center_distance / 7.0, 0, 1)) * 0.5
        score += min(1.0, len(self.discovered_targets) / NUM_TARGETS) * 2.5
        
        self.score_history.pop(0)
        self.score_history.append(score)
        self.leadership_score = sum(self.score_history) / len(self.score_history)
        return self.leadership_score

    def choose_exploration_target(self):
        while True:
            x, y = random.uniform(1, 9), random.uniform(1, 9)
            target = np.array([x, y])
            if not any(np.linalg.norm(np.array(pos) - target) < 1.0 for pos in self.visited_positions):
                return target

    def move_towards_target(self, target):
        direction = target - self.position
        distance = np.linalg.norm(direction)
        if distance > 0:
            self.velocity += (direction / distance) * STEP_SIZE * 1.1

        if self.exploration_target is not None:
            if not self.sensor_line_points:
                self.sensor_line_points = [
                    self.position.copy(),
                    self.position * 0.7 + target * 0.3,
                    self.position * 0.3 + target * 0.7,
                    target.copy()
                ]
            else:
                self.sensor_line_points[0] = self.position.copy()
                mid1 = self.position * (1 - SENSOR_LINE_SMOOTHING) + self.sensor_line_points[1] * SENSOR_LINE_SMOOTHING
                mid2 = mid1 * (1 - SENSOR_LINE_SMOOTHING) + target * SENSOR_LINE_SMOOTHING
                self.sensor_line_points[1] = mid1
                self.sensor_line_points[2] = mid2
                self.sensor_line_points[3] = target.copy()

    def avoid_obstacles(self, obstacles):
        avoidance_force = np.zeros(2, dtype=float)
        was_avoiding = self.avoidance_mode
        self.avoidance_mode = False

        for obs in obstacles:
            direction = self.position - obs.position
            distance = np.linalg.norm(direction)

            if 0 < distance < AVOIDANCE_DISTANCE:
                repulsion_strength = OBSTACLE_REPULSION_STRENGTH * (1 - (distance / AVOIDANCE_DISTANCE) ** 3)
                self.avoidance_mode = True
                
                if distance < AVOIDANCE_DISTANCE * 0.8:
                    repulsion_strength *= 2.0
                
                if distance > 0:
                    avoidance_force += (direction / distance) * repulsion_strength

        next_position = self.position + self.velocity
        for obs in obstacles:
            if np.linalg.norm(next_position - obs.position) < AVOIDANCE_DISTANCE:
                self.velocity *= 0.3
                break

        self.velocity += avoidance_force

        if self.avoidance_mode != was_avoiding:
            self.formation_recovery_timer = FORMATION_RECOVERY_DELAY

    def maintain_formation(self, leader):
        if leader and not self.is_leader:
            target_position = leader.position + self.formation_offset
            formation_error = target_position - self.position
            self.formation_deviation = np.linalg.norm(formation_error)

            if self.formation_recovery_timer > 0:
                self.formation_recovery_timer -= 1
                recovery_factor = 1.0 - (self.formation_recovery_timer / FORMATION_RECOVERY_DELAY)
                formation_force = formation_error * (FORMATION_KEEPING_WEIGHT * 0.2 * recovery_factor)
            elif self.avoidance_mode:
                formation_force = formation_error * (FORMATION_KEEPING_WEIGHT * 0.1)
            elif self.formation_deviation > FORMATION_RECOVERY_THRESHOLD:
                formation_force = formation_error * (FORMATION_KEEPING_WEIGHT * 1.8)
            else:
                formation_force = formation_error * FORMATION_KEEPING_WEIGHT

            self.velocity += formation_force

    def avoid_other_robots(self, robots):
        separation_force = np.zeros(2, dtype=float)
        min_robot_distance = 0.7

        for other in robots:
            if other is self:
                continue

            direction = self.position - other.position
            distance = np.linalg.norm(direction)

            if 0 < distance < min_robot_distance:
                separation_strength = 0.8 * (min_robot_distance - distance) ** 2
                separation_force += (direction / distance) * separation_strength

        self.velocity += separation_force

    def detect_targets(self, all_targets):
        for target in all_targets:
            target_tuple = tuple(target)
            if target_tuple not in self.known_targets:
                distance = np.linalg.norm(self.position - target)
                if distance <= TARGET_DETECTION_RADIUS:
                    self.discovered_targets.add(target_tuple)
                    self.known_targets.add(target_tuple)
                    return target_tuple
        return None

    def share_target_info(self, robots):
        for other in robots:
            if other is not self and np.linalg.norm(self.position - other.position) <= COMMUNICATION_RADIUS:
                other.known_targets.update(self.known_targets)

    def update(self, robots, leader, all_targets, obstacles):
        current_pos = (round(self.position[0], 1), round(self.position[1], 1))
        self.visited_positions.add(current_pos)
        if len(self.visited_positions) > 200:
            self.visited_positions.pop()

        self.detect_targets(all_targets)
        self.share_target_info(robots)

        if self.is_leader:
            all_discovered = set()
            for robot in robots:
                all_discovered.update(robot.discovered_targets)
                
            known_targets = [t for t in all_targets if tuple(t) in self.known_targets 
                             and tuple(t) not in all_discovered]

            if known_targets:
                closest_target = min(known_targets, key=lambda t: np.linalg.norm(self.position - t))
                self.move_towards_target(closest_target)
                self.exploration_target = None
                self.sensor_line_points = []
            else:
                if self.exploration_target is None or self.exploration_time <= 0 or np.linalg.norm(
                        self.position - self.exploration_target) < 0.5:
                    self.exploration_target = self.choose_exploration_target()
                    self.exploration_time = random.randint(30, 50)

                self.exploration_time -= 1
                self.move_towards_target(self.exploration_target)

            for target in all_targets:
                if np.linalg.norm(self.position - target) < TARGET_REACHED_THRESHOLD:
                    self.discovered_targets.add(tuple(target))
                    self.known_targets.add(tuple(target))
                    if self.exploration_target is not None:
                        self.exploration_target = None
                        self.exploration_time = 0
                        self.sensor_line_points = []

        self.avoid_obstacles(obstacles)
        self.avoid_other_robots(robots)
        self.maintain_formation(leader)

        next_position = self.position + self.velocity
        for obs in obstacles:
            if np.linalg.norm(next_position - obs.position) < AVOIDANCE_DISTANCE:
                obs_direction = next_position - obs.position
                if np.linalg.norm(obs_direction) > 0:
                    normalized_direction = obs_direction / np.linalg.norm(obs_direction)
                    safe_position = obs.position + normalized_direction * AVOIDANCE_DISTANCE
                    self.velocity = (safe_position - self.position) * 0.9
                    break

        speed = np.linalg.norm(self.velocity)
        if speed > MAX_SPEED:
            self.velocity = (self.velocity / speed) * MAX_SPEED

        self.position += self.velocity
        self.velocity *= VELOCITY_DECAY

        if self.is_leader and self.exploration_target is not None:
            if np.linalg.norm(self.position - self.exploration_target) < 0.3:
                self.sensor_line_points = []

def update_leadership(robots, obstacles, targets):
    if not targets:
        return None

    for robot in robots:
        robot.calculate_leadership_score(obstacles)

    all_discovered = set()
    for robot in robots:
        all_discovered.update(robot.discovered_targets)

    remaining_targets = [t for t in targets if tuple(t) not in all_discovered]

    if remaining_targets:
        robots_with_target_knowledge = []
        for robot in robots:
            known_undiscovered = [t for t in remaining_targets if tuple(t) in robot.known_targets]
            if known_undiscovered:
                robot.leadership_score += 1.0
                robots_with_target_knowledge.append(robot)

        potential_leaders = robots_with_target_knowledge if robots_with_target_knowledge else robots
    else:
        potential_leaders = robots

    new_leader = max(potential_leaders, key=lambda r: r.leadership_score)

    for robot in robots:
        robot.is_leader = (robot.id == new_leader.id)

    return new_leader

def generate_random_targets(num_targets, obstacles, grid_size=10):
    targets = []
    for _ in range(num_targets):
        while True:
            candidate = np.array([random.uniform(0.5, grid_size - 0.5),
                                 random.uniform(0.5, grid_size - 0.5)])
            min_dist_to_obstacle = min(np.linalg.norm(candidate - obs.position) for obs in obstacles)
            if min_dist_to_obstacle > 1.0:
                targets.append(candidate)
                break
    return targets

def plot_robot_trajectories(robot_positions_history):
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    for robot_id, positions in robot_positions_history.items():
        positions_array = np.array(positions)
        time_steps = np.arange(len(positions_array))
        ax1.plot(time_steps, positions_array[:, 0], label=f'Robot {robot_id} (x)')
        ax2.plot(time_steps, positions_array[:, 1], label=f'Robot {robot_id} (y)')

    ax1.set_title('X-Position vs Time')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('X Position')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Y-Position vs Time')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Y Position')
    ax2.legend()
    ax2.grid(True)

    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')

    for robot_id, positions in robot_positions_history.items():
        positions_array = np.array(positions)
        time_steps = np.arange(len(positions_array))
        ax3d.plot(positions_array[:, 0], positions_array[:, 1], time_steps, label=f'Robot {robot_id}')

    ax3d.set_title('Robot Trajectories in Space-Time')
    ax3d.set_xlabel('X Position')
    ax3d.set_ylabel('Y Position')
    ax3d.set_zlabel('Time Step')
    ax3d.legend()

    plt.tight_layout()
    plt.show()

def draw_visualization(ax, robots, obstacles, targets, leader, trail_history, obstacle_trail_history, 
                       all_discovered_targets, step, leadership_changes, collision_detected):
    ax.clear()
    
    # Draw obstacles and their trails
    for i, obs in enumerate(obstacles):
        if len(obstacle_trail_history[i]) > 1:
            trail = np.array(obstacle_trail_history[i])
            ax.plot(trail[:, 0], trail[:, 1], 'gray', alpha=0.2)
        
        ax.scatter(obs.position[0], obs.position[1], c='black', marker='s', s=150)
        circle = plt.Circle(obs.position, AVOIDANCE_DISTANCE, color='gray', fill=False, linestyle='--', alpha=0.3)
        ax.add_patch(circle)
    
    # Draw robots and their trails
    for i, robot in enumerate(robots):
        if len(trail_history[i]) > 1:
            trail = np.array(trail_history[i])
            ax.plot(trail[:, 0], trail[:, 1], 'b-', alpha=0.3)
        
        color = 'red' if robot.is_leader else 'blue'
        ax.scatter(*robot.position, c=color, s=100)
        ax.text(robot.position[0], robot.position[1] + 0.2, f"R{robot.id}", ha='center', va='center', fontsize=8)
        ax.text(robot.position[0], robot.position[1] - 0.2, f"{robot.leadership_score:.1f}", ha='center', va='center', fontsize=7)
        
        if not robot.is_leader and leader:
            ax.plot([leader.position[0], robot.position[0]], [leader.position[1], robot.position[1]], 'k--', alpha=0.3)
        
        if robot.is_leader and robot.exploration_target is not None:
            ax.scatter(*robot.exploration_target, c='yellow', marker='*', s=100, alpha=0.5)
            
            if robot.sensor_line_points and len(robot.sensor_line_points) >= 4:
                near_target = any(np.linalg.norm(robot.position - target) < TARGET_DETECTION_RADIUS * 1.5 for target in targets)
                
                if not near_target:
                    t = np.linspace(0, 1, 50)
                    points = np.array(robot.sensor_line_points)
                    curve_points = []
                    for ti in t:
                        p = ((1-ti)**3) * points[0] + \
                        3*((1-ti)**2)*ti * points[1] + \
                        3*(1-ti)*(ti**2) * points[2] + \
                        (ti**3) * points[3]
                        curve_points.append(p)
                    curve_points = np.array(curve_points)
                    ax.plot(curve_points[:, 0], curve_points[:, 1], 'y-', alpha=0.7)
        
        detection_circle = plt.Circle(robot.position, TARGET_DETECTION_RADIUS, color='green', fill=False, linestyle=':', alpha=0.2)
        ax.add_patch(detection_circle)
    
    # Draw targets
    for target in targets:
        color = 'green' if tuple(target) in all_discovered_targets else 'red'
        ax.scatter(*target, c=color, marker='X', s=200)
    
    # Mark initial position
    ax.scatter(INITIAL_POSITION[0], INITIAL_POSITION[1], c='purple', marker='^', s=200, label='Initial Position')
    
    # Set plot limits and title
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    title = f"Step {step + 1}: Targets {len(all_discovered_targets)}/{NUM_TARGETS} - Leader: R{leader.id} - Changes: {leadership_changes}"
    if collision_detected:
        title += " - COLLISION DETECTED!"
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)

def visualize_swarm():
    obstacles = [DynamicObstacle(pos) for pos in INITIAL_OBSTACLES]
    targets = generate_random_targets(NUM_TARGETS, obstacles)
    
    offsets = [np.array([0.5, 0.5]), np.array([-0.5, 0.5]), np.array([-0.5, -0.5]), np.array([0.5, -0.5])]
    robots = [Robot(INITIAL_POSITION + offset, offset, i) for i, offset in enumerate(offsets)]
    
    robots[0].is_leader = True
    leader = robots[0]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    trail_history = {i: [] for i in range(NUM_ROBOTS)}
    complete_position_history = {i: [] for i in range(NUM_ROBOTS)}
    obstacle_trail_history = {i: [] for i in range(len(obstacles))}
    max_trail_length = 20
    
    leadership_changes = 0
    collision_detected = False
    all_discovered_targets = set()
    start_time = datetime.datetime.now()
    targets_all_discovered = False
    
    # Main simulation loop
    for step in range(MAX_SIMULATION_STEPS):
        # Update obstacles
        for i, obs in enumerate(obstacles):
            obs.update(obstacles)
            obstacle_trail_history[i].append(obs.position.copy())
            if len(obstacle_trail_history[i]) > max_trail_length:
                obstacle_trail_history[i].pop(0)
        
        # Update leadership
        previous_leader = leader
        new_leader = update_leadership(robots, obstacles, targets)
        if new_leader is not leader:
            leadership_changes += 1
            leader = new_leader
            if previous_leader and new_leader and previous_leader.id != new_leader.id:
                new_leader.exploration_target = None
                new_leader.exploration_time = 0
                new_leader.sensor_line_points = []
        
        # Update all discovered targets
        all_discovered_targets.clear()
        for robot in robots:
            all_discovered_targets.update(robot.discovered_targets)
        
        # Update and track robots
        for i, robot in enumerate(robots):
            robot.update(robots, leader, targets, obstacles)
            complete_position_history[i].append(robot.position.copy())
            
            # Check for collisions
            for obs in obstacles:
                if np.linalg.norm(robot.position - obs.position) < AVOIDANCE_DISTANCE:
                    collision_detected = True
                    print(f"Robot {i} inside avoidance zone at step {step}!")
            
            # Update trail history
            trail_history[i].append(robot.position.copy())
            if len(trail_history[i]) > max_trail_length:
                trail_history[i].pop(0)
        
        # Draw visualization
        draw_visualization(ax, robots, obstacles, targets, leader, trail_history, obstacle_trail_history, 
                           all_discovered_targets, step, leadership_changes, collision_detected)
        
        # Check if all targets are discovered
        if len(all_discovered_targets) == NUM_TARGETS:
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            targets_all_discovered = True
            print(f"All {NUM_TARGETS} targets discovered at step {step} in {duration:.1f} seconds! Stopping simulation.")
            plt.text(5, 5, "All targets discovered!", horizontalalignment='center',
                     fontsize=16, bbox=dict(facecolor='white', alpha=0.8))
            plt.pause(2)
            break
        
        plt.pause(0.1)

    # Return to initial positions if all targets were discovered
    if targets_all_discovered:
        for step in range(MAX_SIMULATION_STEPS):
            ax.clear()
            
            # Update obstacles
            for i, obs in enumerate(obstacles):
                obs.update(obstacles)
                obstacle_trail_history[i].append(obs.position.copy())
                if len(obstacle_trail_history[i]) > max_trail_length:
                    obstacle_trail_history[i].pop(0)
                
                # Draw obstacle
                if len(obstacle_trail_history[i]) > 1:
                    trail = np.array(obstacle_trail_history[i])
                    ax.plot(trail[:, 0], trail[:, 1], 'gray', alpha=0.2)
                
                ax.scatter(obs.position[0], obs.position[1], c='black', marker='s', s=150)
                circle = plt.Circle(obs.position, AVOIDANCE_DISTANCE, color='gray', fill=False, linestyle='--', alpha=0.3)
                ax.add_patch(circle)
            
            # Update robots for return journey
            all_robots_home = True
            for i, robot in enumerate(robots):
                # Move towards initial position
                direction = robot.initial_position - robot.position
                distance = np.linalg.norm(direction)
                if distance > 0:
                    robot.velocity += (direction / distance) * STEP_SIZE * 1.1
                    all_robots_home = False
                
                # Avoid obstacles and other robots
                robot.avoid_obstacles(obstacles)
                robot.avoid_other_robots(robots)
                
                # Limit speed
                speed = np.linalg.norm(robot.velocity)
                if speed > MAX_SPEED:
                    robot.velocity = (robot.velocity / speed) * MAX_SPEED
                
                # Update position
                robot.position += robot.velocity
                complete_position_history[i].append(robot.position.copy())
                robot.velocity *= VELOCITY_DECAY
                
                # Store position history for trail
                trail_history[i].append(robot.position.copy())
                if len(trail_history[i]) > max_trail_length:
                    trail_history[i].pop(0)
                
                # Draw trail
                if len(trail_history[i]) > 1:
                    trail = np.array(trail_history[i])
                    ax.plot(trail[:, 0], trail[:, 1], 'b-', alpha=0.3)
                
                # Draw robot
                color = 'red' if robot.is_leader else 'blue'
                ax.scatter(*robot.position, c=color, s=100)
                ax.text(robot.position[0], robot.position[1] + 0.2, f"R{robot.id}", ha='center', va='center', fontsize=8)
                ax.text(robot.position[0], robot.position[1] - 0.2, f"{robot.leadership_score:.1f}", ha='center', va='center', fontsize=7)
            
            # Draw targets
            for target in targets:
                color = 'green' if tuple(target) in all_discovered_targets else 'red'
                ax.scatter(*target, c=color, marker='X', s=200)
            
            # Mark initial position with a triangle
            ax.scatter(INITIAL_POSITION[0], INITIAL_POSITION[1], c='purple', marker='^', s=200, label='Initial Position')
            
            # Set plot limits and title
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.set_title(f"Returning to initial position: Step {step + 1}")
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Check if all robots have returned home
            if all_robots_home or all(np.linalg.norm(robot.position - robot.initial_position) < 0.1 for robot in robots):
                plt.text(5, 5, "All robots returned home!", horizontalalignment='center',
                         fontsize=16, bbox=dict(facecolor='white', alpha=0.8))
                plt.pause(2)
                break
            
            plt.pause(0.1)
    else:
        # Display final message if simulation ended but not all targets were found
        print(f"Simulation ended after {MAX_SIMULATION_STEPS} steps.")
        print(f"Only {len(all_discovered_targets)}/{NUM_TARGETS} targets were discovered.")
        plt.text(5, 5, f"Simulation timeout: {len(all_discovered_targets)}/{NUM_TARGETS} targets found",
                 horizontalalignment='center', fontsize=16, bbox=dict(facecolor='white', alpha=0.8))
        plt.pause(2)
    
    # Generate position-time graphs
    plot_robot_trajectories(complete_position_history)
    plt.show()

if __name__ == "__main__":
    visualize_swarm()