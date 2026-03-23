import pygame
import random
import math
from OpenGL.GL import *
from OpenGL.GLU import *
import threading
import queue
import json
import ollama
import time

# ---------------- LLM CONFIG ----------------
command_queue = queue.Queue()
current_command = "Waiting for command..."
last_planner_reasoning = "N/A"
last_status_report = "System Nominal"

# ---------------- CONFIG ----------------
WIDTH, HEIGHT = 1100, 750
NUM_SWARM = 14
NUM_FISH = 6

MAX_SPEED = 1.4
MIN_SPEED = 0.3
FISH_SPEED = 1.0

AREA_SIZE = 500

SEABED_Z = 0
ROCK_Z = 20
SWARM_Z = 25

SENSING_RADIUS = 130
SEPARATION_DIST = 35

COHESION_WEIGHT = 0.005
ALIGNMENT_WEIGHT = 0.06
SEPARATION_WEIGHT = 0.18
OBSTACLE_WEIGHT = 0.25
FISH_AVOID_WEIGHT = 0.35

# ---------------- CAMERA ----------------
camera_yaw = 0
camera_pitch = -55
camera_distance = 850

# ---------------- INIT ----------------
pygame.init()
pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF | pygame.OPENGL)
pygame.display.set_caption("Underwater Swarm with Realistic Fish")
clock = pygame.time.Clock()

glEnable(GL_DEPTH_TEST)
glClearColor(0.0, 0.12, 0.22, 1)

glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluPerspective(60, WIDTH / HEIGHT, 1, 5000)
glMatrixMode(GL_MODELVIEW)

# ---------------- ROCKS ----------------
ROCKS = [
    (150, 100, 60),
    (-200, 200, 70),
    (120, -220, 65)
]

# ---------------- MARKERS ----------------
MARKERS = {
    "S": (0, 0),
    "A": (300, -200),
    "B": (300, 250),
    "C": (-300, -200)
}

MARKER_COLORS = {
    "S": (1, 1, 0),
    "A": (0, 1, 0),
    "B": (0, 0.8, 1),
    "C": (1, 0, 1)
}

# ---------------- DYNAMIC FISH ----------------
class Fish:
    def __init__(self):
        self.x = random.uniform(-AREA_SIZE + 100, AREA_SIZE - 100)
        self.y = random.uniform(-AREA_SIZE + 100, AREA_SIZE - 100)
        angle = random.uniform(0, 2 * math.pi)
        self.vx = math.cos(angle) * FISH_SPEED
        self.vy = math.sin(angle) * FISH_SPEED
        self.swim_phase = random.uniform(0, 2 * math.pi)

    def update(self):
        # Smooth wandering
        angle = math.atan2(self.vy, self.vx)
        angle += random.uniform(-0.15, 0.15)

        self.vx = math.cos(angle) * FISH_SPEED
        self.vy = math.sin(angle) * FISH_SPEED

        self.x += self.vx
        self.y += self.vy

        # Boundary bounce
        if abs(self.x) > AREA_SIZE - 50:
            self.vx *= -1
        if abs(self.y) > AREA_SIZE - 50:
            self.vy *= -1

        self.swim_phase += 0.25

    def draw(self):
        glPushMatrix()
        glTranslatef(self.x, self.y, SWARM_Z + 5)

        angle = math.degrees(math.atan2(self.vy, self.vx))
        glRotatef(angle, 0, 0, 1)

        # -------- Body (Ellipse) --------
        glColor3f(0.2, 0.6, 1.0)
        glBegin(GL_POLYGON)
        for i in range(30):
            theta = 2 * math.pi * i / 30
            x = 14 * math.cos(theta)
            y = 8 * math.sin(theta)
            glVertex3f(x, y, 0)
        glEnd()

        # -------- Tail (Animated) --------
        tail_wave = math.sin(self.swim_phase) * 4

        glColor3f(0.1, 0.4, 0.8)
        glBegin(GL_TRIANGLES)
        glVertex3f(-14, 0, 0)
        glVertex3f(-24, 8 + tail_wave, 0)
        glVertex3f(-24, -8 - tail_wave, 0)
        glEnd()

        # -------- Top Fin --------
        glColor3f(0.15, 0.5, 0.9)
        glBegin(GL_TRIANGLES)
        glVertex3f(-2, 6, 0)
        glVertex3f(4, 14, 0)
        glVertex3f(8, 6, 0)
        glEnd()

        # -------- Eye --------
        glColor3f(0, 0, 0)
        glBegin(GL_POLYGON)
        for i in range(12):
            theta = 2 * math.pi * i / 12
            x = 6 + 2 * math.cos(theta)
            y = 2 * math.sin(theta)
            glVertex3f(x, y, 1)
        glEnd()

        glPopMatrix()

# ---------------- SWARM ROBOT ----------------
class SwarmRobot:
    def __init__(self, group_id=0):
        self.x = random.uniform(-250, 250)
        self.y = random.uniform(-250, 250)
        angle = random.uniform(0, 2 * math.pi)
        self.vx = math.cos(angle)
        self.vy = math.sin(angle)
        self.group_id = group_id
        
        self.is_stuck = False
        self.stuck_timer = 0



    def update(self, swarm, fishes):

        neighbors = []
        for r in swarm:
            if r is self:
                continue
            dist = math.hypot(self.x - r.x, self.y - r.y)
            if dist < SENSING_RADIUS:
                neighbors.append(r)

        # Separation (Avoid everyone, but more strongly different groups)
        sep_x, sep_y = 0, 0
        for r in neighbors:
            dx = self.x - r.x
            dy = self.y - r.y
            dist = math.hypot(dx, dy)
            if 0 < dist < SEPARATION_DIST:
                # Stronger repulsion if different group
                weight = 2.0 if self.group_id != r.group_id else 1.0
                sep_x += (dx / dist) * weight
                sep_y += (dy / dist) * weight

        # Defer separation application to allow target priority scaling

        # Alignment & Cohesion (Only with SAME GROUP)
        same_group_neighbors = [r for r in neighbors if r.group_id == self.group_id]
        
        if same_group_neighbors:
            avg_vx = sum(r.vx for r in same_group_neighbors) / len(same_group_neighbors)
            avg_vy = sum(r.vy for r in same_group_neighbors) / len(same_group_neighbors)

            # Apply proximity factor scaling to swarm rules further down
            self.vx += (avg_vx - self.vx) * ALIGNMENT_WEIGHT
            self.vy += (avg_vy - self.vy) * ALIGNMENT_WEIGHT

            center_x = sum(r.x for r in same_group_neighbors) / len(same_group_neighbors)
            center_y = sum(r.y for r in same_group_neighbors) / len(same_group_neighbors)

            if len(GROUP_TARGETS) == 0:
                self.vx += (center_x - self.x) * COHESION_WEIGHT
                self.vy += (center_y - self.y) * COHESION_WEIGHT

        # Group Target Attraction (if exists)
        has_target = self.group_id in GROUP_TARGETS
        if has_target:
            tx, ty = GROUP_TARGETS[self.group_id]
            dx = tx - self.x
            dy = ty - self.y
            dist = math.hypot(dx, dy)
            if dist > 0:
                # Stronger pull toward target; gentle slow-down near arrival
                pull_strength = 0.55
                self.vx += (dx / dist) * pull_strength
                self.vy += (dy / dist) * pull_strength
                if dist < 45:
                    self.vx *= 0.8
                    self.vy *= 0.8

        # Apply separation; soften slightly when targeting
        sep_scale = SEPARATION_WEIGHT * (0.6 if has_target else 1.0)
        self.vx += sep_x * sep_scale
        self.vy += sep_y * sep_scale

        # Rock Avoidance
        for rx, ry, size in ROCKS:
            dx = self.x - rx
            dy = self.y - ry
            dist = math.hypot(dx, dy)
            if dist < size + 90:
                self.vx += (dx / dist) * OBSTACLE_WEIGHT
                self.vy += (dy / dist) * OBSTACLE_WEIGHT

        # Fish Avoidance
        for fish in fishes:
            dx = self.x - fish.x
            dy = self.y - fish.y
            dist = math.hypot(dx, dy)
            if dist < 80:
                self.vx += (dx / dist) * FISH_AVOID_WEIGHT
                self.vy += (dy / dist) * FISH_AVOID_WEIGHT

        # Soft Boundary
        margin = 120
        turn = 0.4

        if self.x < -AREA_SIZE + margin:
            self.vx += turn
        if self.x > AREA_SIZE - margin:
            self.vx -= turn
        if self.y < -AREA_SIZE + margin:
            self.vy += turn
        if self.y > AREA_SIZE - margin:
            self.vy -= turn

        # Speed Control
        speed = math.hypot(self.vx, self.vy)

        if speed > MAX_SPEED:
            self.vx = (self.vx / speed) * MAX_SPEED
            self.vy = (self.vy / speed) * MAX_SPEED

        if speed < MIN_SPEED:
            angle = random.uniform(0, 2 * math.pi)
            self.vx += math.cos(angle) * 0.2
            self.vy += math.sin(angle) * 0.2

        self.vx *= 0.995
        self.vy *= 0.995

        self.x += self.vx
        self.y += self.vy

        # Stuck Detection (Clear if moving again)
        speed = math.hypot(self.vx, self.vy)
        
        at_target = False
        if has_target:
            tx, ty = GROUP_TARGETS[self.group_id]
            if math.hypot(tx - self.x, ty - self.y) < 50:
                at_target = True

        if speed < 0.05 and not at_target:
            self.stuck_timer += 1
        else:
            self.stuck_timer = 0
            self.is_stuck = False # Recover if moving
            
        if self.stuck_timer > 120: # 2 seconds stuck
             self.is_stuck = True

    def draw(self):
        glPushMatrix()
        glTranslatef(self.x, self.y, SWARM_Z)

        angle = math.degrees(math.atan2(self.vx, self.vy))
        glRotatef(angle, 0, 0, 1)

        # Color based on Status/Group
        if self.is_stuck:
             glColor3f(0.5, 0.5, 0.5) # Gray (Stuck)
        else:
            colors = [(0, 0, 0), (1, 1, 1), (0, 0, 1), (0, 1, 0), (1, 1, 0)]
            c = colors[self.group_id % len(colors)]
            glColor3f(*c)
        
        glBegin(GL_TRIANGLES)

        glVertex3f(0, 15, 0)
        glVertex3f(-7, -7, 0)
        glVertex3f(7, -7, 0)
        glEnd()

        glPopMatrix()

# ---------------- DRAW FUNCTIONS ----------------
def draw_arena():
    glColor3f(1, 1, 1)
    glLineWidth(3)
    glBegin(GL_LINE_LOOP)
    glVertex3f(-AREA_SIZE, -AREA_SIZE, 5)
    glVertex3f(AREA_SIZE, -AREA_SIZE, 5)
    glVertex3f(AREA_SIZE, AREA_SIZE, 5)
    glVertex3f(-AREA_SIZE, AREA_SIZE, 5)
    glEnd()

def draw_floor():
    glColor3f(0.1, 0.4, 0.3)
    glBegin(GL_QUADS)
    glVertex3f(-2000, -2000, 0)
    glVertex3f(2000, -2000, 0)
    glVertex3f(2000, 2000, 0)
    glVertex3f(-2000, 2000, 0)
    glEnd()

def draw_rock(x, y, size):
    glPushMatrix()
    glTranslatef(x, y, ROCK_Z)
    glColor3f(0.55, 0.45, 0.3)
    quad = gluNewQuadric()
    gluSphere(quad, size, 40, 40)
    gluDeleteQuadric(quad)
    glPopMatrix()

def draw_marker(x, y, color):
    glPushMatrix()
    glTranslatef(x, y, SWARM_Z)
    quad = gluNewQuadric()

    glColor3f(*color)
    gluSphere(quad, 25, 30, 30)

    glColor3f(0.9, 0.9, 0.9)
    glTranslatef(0, 0, 25)
    gluCylinder(quad, 5, 5, 70, 15, 15)

    gluDeleteQuadric(quad)
    glPopMatrix()

# ---------------- COUNCIL OF AGENTS ----------------

def monitor_swarm(swarm):
    """Monitor Agent: Diagnoses swarm status."""
    report = []
    stuck_count = sum(1 for r in swarm if r.is_stuck)
    
    if stuck_count > 0:
        report.append(f"WARNING: {stuck_count} robots immobile/stuck")
        
    return "; ".join(report) if report else "Nominal"

def validate_safety(user_input):
    """Safety Validator: Checks for unsafe keywords/bounds."""
    unsafe_keywords = ["crash", "explode", "kill", "infinite", "nan"]
    if any(k in user_input.lower() for k in unsafe_keywords):
        return False, "Safety Violation: Harmful command rejected."
    if len(user_input) > 200:
        return False, "Safety Violation: Input too long (buffer overflow prevention)."
    return True, "Safe"

def parse_json_from_content(content):
    """Extract the first JSON object from model output, ignoring extra text."""
    decoder = json.JSONDecoder()
    cleaned = content.replace("```json", "").replace("```", "")
    # Remove line comments which are invalid in JSON
    cleaned = "\n".join(line.split("//")[0] for line in cleaned.splitlines())
    start = cleaned.find("{")
    if start == -1:
        return None
    try:
        obj, _ = decoder.raw_decode(cleaned[start:])
        return obj
    except json.JSONDecodeError:
        return None

# ---------------- LLM LISTENERS ----------------

def llm_listener():
    global current_command, last_planner_reasoning, last_status_report
    print("COUNCIL OF AGENTS ONLINE. Awaiting Orders:")
    
    # --- PROMPTS ---
    PLANNER_PROMPT = """You are the STRATEGIC COMMANDER of an underwater swarm.
    Your goal is to interpret the User's Intent and the Swarm's Status.
    
    Output a STRATEGY in natural language.
    - If status is critical (e.g., stuck), prioritize recovery.
    - If user wants to split, explain how (e.g., "Divide into 2 groups, send Group 0 to A...").
    - If user wants to go to a sequence of points (e.g., "go to S through B"), specify forming a path (e.g., "Move via path B, then S").
    - If user wants to attack/move to a single point, map it to the markers (S, A, B, C).
    
    Keep it concise (1-2 sentences). Do NOT output JSON."""

    EXECUTOR_PROMPT = """You are the TACTICAL PILOT.
    Convert the Commander's Strategy into rigid JSON parameters for the swarm control system.
    
    Available parameters:
    - action: "split", "regroup", "merge", "stop", "go", "path"
    - split_n: int (1-5)
    - target_marker: string ("S", "A", "B", "C"), list of strings (["A", "B"]), or dict mapping markers to group IDs
    - speed_factor: 0.0 to 5.0
    - separation: 0.1 to 1.0 (higher = disperse)
    - cohesion: 0.0 to 0.1 (higher = tight ball)
    
    NOTE: For the "path" action (moving sequentially through waypoints), use `target_marker` as an ordered list of markers (e.g., ["B", "S"]).
    
    Output ONLY valid JSON."""

    while True:
        try:
            user_input = input(">> ")
            if not user_input: continue
            
            # --- PHASE 1: SAFETY VALIDATOR ---
            is_safe, reason = validate_safety(user_input)
            if not is_safe:
                print(f"[VALIDATOR] BLOCK: {reason}")
                current_command = f"BLOCKED: {reason}"
                continue

            current_command = f"Processing: {user_input}"
            
            # --- PHASE 2: MONITOR AGENT ---
            swarm_status = monitor_swarm(swarm)
            last_status_report = swarm_status
            print(f"[MONITOR] Status: {swarm_status}")

            # --- PHASE 3: PLANNER AGENT ---
            planner_input = f"User Order: '{user_input}'. Swarm Status: '{swarm_status}'"
            
            plan_response = ollama.chat(model='llama3.2', messages=[
                {'role': 'system', 'content': PLANNER_PROMPT},
                {'role': 'user', 'content': planner_input},
            ])
            
            strategy = plan_response['message']['content']
            last_planner_reasoning = strategy
            print(f"[PLANNER] Strategy: {strategy}")
            
            # --- PHASE 4: EXECUTOR AGENT ---
            exec_response = ollama.chat(model='llama3.2', messages=[
                {'role': 'system', 'content': EXECUTOR_PROMPT},
                {'role': 'user', 'content': f"Execute this strategy: {strategy}"},
            ])
            
            content = exec_response['message']['content']
            
            params = parse_json_from_content(content)
            if params is not None:
                command_queue.put(params)
                current_command = f"Executed: {user_input}"
                print(f"[EXECUTOR] Parameters: {params}")
            else:
                current_command = "Error: Executor failed to generate JSON"
                print(f"[EXECUTOR] Error: {content}")
                
        except Exception as e:
            print(f"COUNCIL Error: {e}")
            current_command = f"Error: {e}"

# ---------------- MAIN ----------------
GROUP_TARGETS = {}
WANDER_GROUP_TARGETS = True
ACTIVE_WAYPOINTS = []

swarm = [SwarmRobot(0) for _ in range(NUM_SWARM)]
fishes = [Fish() for _ in range(NUM_FISH)]

# Start LLM Thread
threading.Thread(target=llm_listener, daemon=True).start()

running = True
font = pygame.font.SysFont('Arial', 24)

while running:
    clock.tick(60)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Process LLM Commands
    while not command_queue.empty():
        params = command_queue.get()
        print(f"Applying parameters: {params}")
        
        if "cohesion" in params: COHESION_WEIGHT = params["cohesion"]
        if "alignment" in params: ALIGNMENT_WEIGHT = params["alignment"]
        if "separation" in params: SEPARATION_WEIGHT = params["separation"]
        if "obstacle_avoidance" in params: OBSTACLE_WEIGHT = params["obstacle_avoidance"]
        if "fish_avoidance" in params: FISH_AVOID_WEIGHT = params["fish_avoidance"]
        if "speed_factor" in params: MAX_SPEED = 1.4 * params["speed_factor"]
        
        # Action Handler
        action = params.get("action", "").lower()
        
        if action == "split":
            try:
                n = int(params.get("split_n", 2))
                n = max(1, min(n, 5)) # Limit to 5 max for sanity
                
                # Assign groups
                for i, robot in enumerate(swarm):
                    robot.group_id = i % n
                
                # Set initial random targets for groups to force them apart
                GROUP_TARGETS.clear()
                for i in range(n):
                    # Pick random points far from center to start
                    angle = (2 * math.pi / n) * i
                    tx = math.cos(angle) * 300
                    ty = math.sin(angle) * 300
                    GROUP_TARGETS[i] = [tx, ty] # Use list for mutability
                    
                SEPARATION_WEIGHT = 0.5 # Force push apart
                WANDER_GROUP_TARGETS = True
                
            except:
                pass
                
        elif action in ["regroup", "merge"]:
            # Reset everyone to group 0 and unstick them
            for robot in swarm:
                robot.group_id = 0
                robot.is_stuck = False
                robot.stuck_timer = 0
            GROUP_TARGETS.clear()
            SEPARATION_WEIGHT = 0.18 # Reset to normal
            MAX_SPEED = max(MAX_SPEED, 1.4) # Ensure they can move
            WANDER_GROUP_TARGETS = True
            ACTIVE_WAYPOINTS = []
            
        elif action == "stop":
            GROUP_TARGETS.clear()
            MAX_SPEED = 0.05 # Verify slow crawl
            WANDER_GROUP_TARGETS = True
            ACTIVE_WAYPOINTS = []
            
        # Wandering Logic for Groups
        if WANDER_GROUP_TARGETS:
            for gid in list(GROUP_TARGETS.keys()):
                # Keep targets mostly still to prevent drift, only minor flutter
                GROUP_TARGETS[gid][0] += random.uniform(-0.2, 0.2)
                GROUP_TARGETS[gid][1] += random.uniform(-0.2, 0.2)
                # Keep in bounds
                GROUP_TARGETS[gid][0] = max(-AREA_SIZE, min(AREA_SIZE, GROUP_TARGETS[gid][0]))
                GROUP_TARGETS[gid][1] = max(-AREA_SIZE, min(AREA_SIZE, GROUP_TARGETS[gid][1]))

        # Handle Target
        if "target_marker" in params:
            tm = params["target_marker"]
            if action == "path":
                if isinstance(tm, str):
                    tm = [p.strip().upper() for p in tm.replace('.', ',').split(',')]
                if isinstance(tm, list) and len(tm) > 0:
                    valid = [m.upper() for m in tm if isinstance(m, str) and m.upper() in MARKERS]
                    if valid:
                        ACTIVE_WAYPOINTS = valid
                        current_wp = ACTIVE_WAYPOINTS[0]
                        t_pos = MARKERS[current_wp]
                        for robot in swarm:
                            robot.group_id = 0
                            if hasattr(robot, 'is_stuck'):
                                robot.is_stuck = False
                            if hasattr(robot, 'stuck_timer'):
                                robot.stuck_timer = 0
                        GROUP_TARGETS.clear()
                        for gid in range(max(5, len(swarm))):
                            GROUP_TARGETS[gid] = [t_pos[0] + random.uniform(-30, 30), t_pos[1] + random.uniform(-30, 30)]
                        WANDER_GROUP_TARGETS = False
            elif isinstance(tm, dict):
                ACTIVE_WAYPOINTS = []
                for marker, payload in tm.items():
                    marker = marker.upper()
                    if marker not in MARKERS:
                        continue
                    group_id = None
                    if isinstance(payload, dict):
                        group_id = payload.get("group_id")
                    elif isinstance(payload, int):
                        group_id = payload
                    if group_id is None:
                        continue
                    t_pos = MARKERS[marker]
                    GROUP_TARGETS[int(group_id)] = [
                        t_pos[0] + random.uniform(-30, 30),
                        t_pos[1] + random.uniform(-30, 30)
                    ]
                WANDER_GROUP_TARGETS = False
            else:
                ACTIVE_WAYPOINTS = []
                if isinstance(tm, str):
                    # In case LLM returns "a.b.c.s" or "A, B"
                    tm = [p.strip().upper() for p in tm.replace('.', ',').split(',')]
                    
                if isinstance(tm, list) and len(tm) > 0:
                    valid = [m.upper() for m in tm if isinstance(m, str) and m.upper() in MARKERS]
                    if valid:
                        WANDER_GROUP_TARGETS = False
                        if len(valid) == 1:
                            t_pos = MARKERS[valid[0]]
                            for gid in range(max(5, len(swarm))):
                                GROUP_TARGETS[gid] = [t_pos[0] + random.uniform(-50, 50), t_pos[1] + random.uniform(-50, 50)]
                        else:
                            for gid in range(max(5, len(swarm))):
                                assigned_marker = valid[gid % len(valid)]
                                t_pos = MARKERS[assigned_marker]
                                GROUP_TARGETS[gid] = [t_pos[0] + random.uniform(-30, 30), t_pos[1] + random.uniform(-30, 30)]

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    glLoadIdentity()
    glTranslatef(0, 0, -camera_distance)
    glRotatef(camera_pitch, 1, 0, 0)
    glRotatef(camera_yaw, 0, 1, 0)

    draw_floor()
    draw_arena()

    for rx, ry, size in ROCKS:
        draw_rock(rx, ry, size)

    for name, pos in MARKERS.items():
        draw_marker(pos[0], pos[1], MARKER_COLORS[name])

    for fish in fishes:
        fish.update()
        fish.draw()

    for robot in swarm:
        robot.update(swarm, fishes)
        robot.draw()

    if ACTIVE_WAYPOINTS:
        current_wp = ACTIVE_WAYPOINTS[0]
        t_pos = MARKERS[current_wp]
        reached = sum(1 for robot in swarm if math.hypot(t_pos[0] - robot.x, t_pos[1] - robot.y) < 70)
        
        if reached > len(swarm) * 0.6:  # 60% of swarm reached the waypoint
            ACTIVE_WAYPOINTS.pop(0)
            if ACTIVE_WAYPOINTS:
                next_wp = ACTIVE_WAYPOINTS[0]
                t_pos = MARKERS[next_wp]
                current_command = f"Reached {current_wp}, proceeding to {next_wp}"
                for gid in range(max(5, len(swarm))):
                    GROUP_TARGETS[gid] = [t_pos[0] + random.uniform(-40, 40), t_pos[1] + random.uniform(-40, 40)]
            else:
                current_command = f"Reached final destination {current_wp}"
                # Keep target at final destination
                for gid in range(max(5, len(swarm))):
                    GROUP_TARGETS[gid] = [t_pos[0] + random.uniform(-40, 40), t_pos[1] + random.uniform(-40, 40)]

    # Capture matrices for projection
    model_view = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)
    viewport = glGetIntegerv(GL_VIEWPORT)
    
    marker_screen_coords = []
    for name, pos in MARKERS.items():
        # Project 3D (x, y, z) to 2D (x, y)
        # We add some Z to make it float above the marker
        wx, wy, wz = gluProject(pos[0], pos[1], SWARM_Z + 60, model_view, projection, viewport)
        # OpenGL y is from bottom, so it works with glWindowPos2d
        marker_screen_coords.append((name, wx, wy))


    # Draw Text (2D Overlay)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
    
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    
    text_surface = font.render(current_command, True, (255, 255, 255, 255))
    text_data = pygame.image.tostring(text_surface, "RGBA", True)
    glWindowPos2d(10, HEIGHT - 40)
    glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

    # Draw Marker Labels
    for name, mx, my in marker_screen_coords:
        label_surface = font.render(name, True, (255, 255, 0, 255))
        label_data = pygame.image.tostring(label_surface, "RGBA", True)
        glWindowPos2d(mx, my)
        glDrawPixels(label_surface.get_width(), label_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, label_data)


    glPopMatrix()
    
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)

    pygame.display.flip()

pygame.quit()
