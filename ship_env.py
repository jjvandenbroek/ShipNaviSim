import gymnasium as gym
import pygame
from datetime import timedelta
import numpy as np
import polars as pl
import utils
import minari
from collections import deque
from pygame import gfxdraw

class ShipEnvironment(gym.Env):
    """
    Gymnasium environment for ship navigation based on real AIS trajectory data.
    
    ============================================================================
    EPISODE GENERATION OVERVIEW - HOW DATASET EPISODES ARE CREATED:
    ============================================================================
    
    The episodes in your dataset come from REAL RECORDED SHIP TRAJECTORIES.
    Each episode represents one ship's complete journey from its starting position
    to its destination, extracted from historical AIS (Automatic Identification System) data.
    
    STEP 1: DATA PREPROCESSING (chunk_to_traj function)
    ---------------------------------------------------
    Raw AIS data → Clean, interpolated trajectories
    
    - Input: Sparse, irregular AIS position reports (lat/lon, speed, heading)
    - Process:
      * Convert geographic coordinates (lat/lon) to local Cartesian (x/y meters)
      * Convert units (knots→m/s, compass degrees→radians)
      * Interpolate onto regular time grid (e.g., every 10 seconds)
      * Use Haversine distance for accurate lat/lon→meter conversion
    - Output: Array of states [x, y, speed, heading] at regular intervals
    
    STEP 2: EPISODE INITIALIZATION (reset method)
    ----------------------------------------------
    Select one ship trajectory to become an "episode"
    
    - Choose ego ship: Select one trajectory as the agent to control/observe
    - Set starting state: First recorded position becomes initial state
    - Set goal: Last recorded position becomes target destination
    - Identify neighbors: Find all ships whose trajectories overlap in time
    - Generate expert actions: Calculate (dx, dy, dheading) from consecutive states
    - Validate: Reject trajectories with erratic movements (too large jumps/turns)
    
    STEP 3: EXPERT ACTION EXTRACTION (gen_actions_from_data, infer_action_from_trajectory)
    ---------------------------------------------------------------------------------------
    Convert recorded trajectory into sequence of actions
    
    For each consecutive pair of states (t, t+1):
    - Calculate: dx = x[t+1] - x[t], dy = y[t+1] - y[t]
    - Calculate: dheading = heading[t+1] - heading[t] (shortest angle)
    - Validate: Ensure changes are within physical bounds:
      * |dx|, |dy| < 5% of environment size (not teleporting)
      * |dheading| < 90° (no sudden spin)
    - If any action invalid → reject entire trajectory
    - Scale to action space range (e.g., [-1, 1] if normalized)
    
    STEP 4: OBSERVATION CONSTRUCTION (_get_observation method)
    -----------------------------------------------------------
    Build observation showing ego state + nearby ships
    
    At each timestep:
    - Ego observation: Sliding window of last N states [x, y, speed, heading, drift]
    - Find neighbors: Query all ships active at current timestamp
    - Calculate distances: Euclidean distance from ego to each neighbor
    - Select nearest N: Sort by distance, keep closest neighbors
    - Extract histories: For each neighbor, get their last N states
    - Pad if needed: If fewer than N neighbors, pad with zeros
    - Optional: Transform to ego's frame of reference (relative positions)
    
    STEP 5: EPISODE ROLLOUT (create_minari_dataset function)
    ---------------------------------------------------------
    Execute trajectory and record transitions
    
    For each valid ship trajectory:
    - Reset environment with this ship as ego
    - Loop through pre-computed expert actions:
      * Execute action in environment
      * Record transition: (observation, action, reward, next_observation)
      * Update ego state according to action
      * Update neighbor observations (they follow their real trajectories)
      * Check termination: goal reached or boundary violated
    - Save episode to Minari dataset
    
    ============================================================================
    KEY INSIGHT: EXPERT DEMONSTRATIONS
    ============================================================================
    
    Your dataset contains IMITATION LEARNING data:
    - States: What the ship observed (own state + neighbors)
    - Actions: What the real ship captain actually did
    - This is "expert behavior" extracted from real navigation data
    - Agents can learn by imitating these expert demonstrations (Behavioral Cloning)
    
    WHAT MAKES AN EPISODE "VALID":
    - Complete trajectory (start to end)
    - Smooth movements (no erratic jumps)
    - Within bounds (stays in region of interest)
    - Physically plausible (speed/turn changes are realistic)
    
    ============================================================================
    """
    def __init__(self, ship_trajectories, ship_times, overlap_idx, region_of_interest,
                 ego_pos: int = 0, observation_history_length: int = 10, n_neighbor_agents: int = 5, render_histL: int = 1000,
                 normalize_xy: bool = False, max_steps: int = 1000, second_perts = 10, use_time_fea=False,
                 drop_neighbor=False, use_dis_fea=False, use_drift_fea=False, use_FoR = False, scale_act=False):
        """
        Initialize the ship navigation environment with preprocessed trajectory data.
        
        KEY DATA STRUCTURES FOR EPISODE GENERATION:
        
        ship_trajectories: List[np.ndarray]
            - One array per ship containing its complete trajectory
            - Each trajectory: array of shape (n_timesteps, 4 or 5)
            - Columns: [x_meters, y_meters, speed_m/s, heading_radians, (drift_radians)]
            - These are the "expert demonstrations" - real recorded ship movements
            - Episodes are generated by selecting one trajectory as ego agent
        
        ship_times: List[np.ndarray]
            - One array per ship with Unix timestamps for each state
            - Parallel to ship_trajectories: ship_times[i][j] is time of ship_trajectories[i][j]
            - Used to synchronize observations (which ships are visible when)
        
        overlap_idx: Dict[int, List[int]]
            - Maps each ship_id to list of ships that overlap in time
            - overlap_idx[ego_id] gives all ships that could be neighbors in that episode
            - Pre-computed for efficiency (avoids checking all ships every step)
        
        region_of_interest: Dict
            - Geographic bounds: {'LON': [min_lon, max_lon], 'LAT': [min_lat, max_lat]}
            - Defines the navigation area in lat/lon coordinates
            - Converted to meters: origin at southwest corner
        
        EPISODE PARAMETERS:
        
        ego_pos: Which ship trajectory to use as the ego agent (index into ship_trajectories)
        observation_history_length: How many past timesteps to include in observation
        n_neighbor_agents: Maximum number of neighboring ships to observe
        second_perts: Time interval between trajectory samples (seconds per timestep)
        
        OBSERVATION/ACTION SPACE PARAMETERS:
        
        normalize_xy: Whether to normalize positions to [-1, 1] range
        scale_act: Whether actions are in [-1, 1] (True) or raw meters/radians (False)
        use_drift_fea: Include drift angle (heading vs movement direction) in observations
        use_FoR: Transform neighbor positions to ego's frame of reference
        use_dis_fea: Include distances to neighbors in observation
        use_time_fea: Include time encoding in observation
        drop_neighbor: Exclude neighbor observations (ego-only navigation)
        """
        super(ShipEnvironment, self).__init__()
        
        # Store preprocessed trajectory data - these ARE the episodes
        self.ship_trajectories = ship_trajectories  # The expert demonstrations
        self.ship_times = ship_times                # Timestamps for temporal alignment
        self.overlap_idx = overlap_idx              # Which ships can be neighbors
        self.region_of_interest = region_of_interest
        self.ego_pos = ego_pos
        self.observation_history_length = observation_history_length
        self.n_neighbor_agents = n_neighbor_agents
        self.normalize_xy = normalize_xy
        self.max_steps = max_steps
        self.second_perts = second_perts #seconds per timestep
        self.CPD = 0 #Closest Point Distance
        self.render_histL = render_histL
        self.nearmiss_threshold = 555
                        
        # Calculate the dimensions of the area in meters
        origin_lon, origin_lat = region_of_interest['LON'][0], region_of_interest['LAT'][0]
        # self.max_x = region_of_interest['LON'][1] - origin_lon
        # self.max_y = region_of_interest['LAT'][1] - origin_lat
        self.max_x = utils.haversine_distance(origin_lon, origin_lat, region_of_interest['LON'][1], origin_lat)
        self.max_y = utils.haversine_distance(origin_lon, origin_lat, origin_lon, region_of_interest['LAT'][1])        
        
        # Define action space (dx, dy, dheading)        
        if scale_act is False:
            self.action_space = gym.spaces.Box(
                low=np.array([-self.max_x/20, -self.max_y/20, -np.pi/2]),
                high=np.array([self.max_x/20, self.max_y/20, np.pi/2]),
                dtype=np.float32
            )
        else:
            self.action_space = gym.spaces.Box(
                low=np.array([-1, -1, -1]),
                high=np.array([1, 1, 1]),
                dtype=np.float32
            )
        
        self.features_per_agent = 4  # x, y, speed, heading
        if use_drift_fea:
            self.features_per_agent += 1 # x, y, speed, heading, drift
        # Define observation space
        ego_obs_shape = (self.observation_history_length + 1, self.features_per_agent)
        neighbor_obs_shape = (self.n_neighbor_agents, self.observation_history_length + 1, self.features_per_agent)

        if self.normalize_xy:
            obs_low = np.array([-1, -1, 0, 0], dtype=np.float32)
            obs_high = np.array([1, 1, 1000, 2*np.pi], dtype=np.float32)
        else:
            obs_low = np.array([0, 0, 0, 0], dtype=np.float32)
            obs_high = np.array([self.max_x, self.max_y, 1000, 2*np.pi], dtype=np.float32)
            if use_drift_fea:
                obs_low = np.array([0, 0, 0, 0, -2*np.pi], dtype=np.float32)
                obs_high = np.array([self.max_x, self.max_y, 1000, 2*np.pi, 2*np.pi], dtype=np.float32)
        
        self.padded_val = np.array([0, 0, 0, 0], dtype=np.float32)        
        if use_drift_fea:
            self.padded_val = np.array([0, 0, 0, 0, 0], dtype=np.float32)        
        obs_space_dict = {
            'ego': gym.spaces.Box(
                low=np.broadcast_to(obs_low,  ego_obs_shape),
                high=np.broadcast_to(obs_high,  ego_obs_shape),
                shape=ego_obs_shape,
                dtype=np.float32
            ),
            'neighbors': gym.spaces.Box(
                low=np.broadcast_to(obs_low,  neighbor_obs_shape),
                high=np.broadcast_to(obs_high,  neighbor_obs_shape),
                shape=neighbor_obs_shape,
                dtype=np.float32
            ),
            'nearest_dis':gym.spaces.Box(
                low=0,
                high=np.sqrt(self.max_x**2+self.max_y**2),
                shape=(self.n_neighbor_agents,),
                dtype=np.float32
            ),
            'time': gym.spaces.Box(low=np.array([-1]*8), high=np.array([1]*8), dtype=np.float32),
            "goal":gym.spaces.Box(
                low=obs_low[:2],
                high=obs_high[:2],
                dtype=np.float32
            ),                
        }
        if use_FoR:
            #Due to rotating we can't control
            new_obs_low = np.array([-np.inf, -np.inf, 0, 0], dtype=np.float32)
            new_obs_high = np.array([np.inf, np.inf, 1000, 2*np.pi], dtype=np.float32)
            obs_space_dict['neighbors'] = gym.spaces.Box(
                low=np.broadcast_to(new_obs_low,  neighbor_obs_shape),
                high=np.broadcast_to(new_obs_high,  neighbor_obs_shape),
                shape=neighbor_obs_shape,
                dtype=np.float32
            )
        self.use_drift_fea = use_drift_fea
        self.drop_neighbor = drop_neighbor
        self.use_dis_fea = use_dis_fea
        self.use_FoR = use_FoR
        self.use_time_fea = use_time_fea
        self.scale_act = scale_act
        if drop_neighbor:
            obs_space_dict.pop("neighbors", None)
        if use_dis_fea is False:
            obs_space_dict.pop("nearest_dis", None)
        if use_time_fea is False:
            obs_space_dict.pop("time", None)
        self.observation_space = gym.spaces.Dict(obs_space_dict)
        
        self.goal_threshold = 200  # meters, can be adjusted
        self.max_dx = 0
        self.max_dy = 0
        self.count_dx = []
        # Pygame initialization
        self.render_mode = "rgb_array1"
        pygame.init()
        self.screen_width = 1500
        self.screen_height = 1500
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.font = pygame.font.Font(None, 36)         
        star_img = pygame.image.load('./star-icon.jpg')
        self.star_img = pygame.transform.scale(star_img, (30, 30))  # Adjust size as needed
    
    def render(self):
        self.screen.fill((255, 255, 255))  # Fill screen with white

        # Calculate scaling factors
        scale_x = self.screen_width / self.max_x
        scale_y = self.screen_height / self.max_y

        # Draw ego ship
        ego_x, ego_y = self.current_state['x'], self.current_state['y']
        ego_heading = self.current_state['heading']
        if(len(self.save_past_ego) != 0):                  
            ego_heading = utils.cal_course(self.save_past_ego[0][0],self.save_past_ego[0][1], ego_x, ego_y, ego_heading)
        
        if self.normalize_xy:
            ego_x = (ego_x + 1) * self.max_x / 2
            ego_y = (ego_y + 1) * self.max_y / 2
        
        screen_x = int(ego_x * scale_x)
        screen_y = int(self.screen_height - ego_y * scale_y)  # Flip y-coordinate
        
        # Draw custom ego ship
        cartwidth = 25.0
        cartheight = 10.0
        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r+(r-l)/4, (t+b)/2), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-ego_heading)  # Note the negative sign
            coord = (coord[0] + screen_x, coord[1] + screen_y)
            pole_coords.append(coord)
    
        ship_color = (255, 0, 0)  # Red for ego ship
        gfxdraw.aapolygon(self.screen, pole_coords, ship_color)
        gfxdraw.filled_polygon(self.screen, pole_coords, ship_color)
        
        # Draw ego ship's history
        ego_obs = self.current_obs['ego']
        # for i in range(len(ego_obs) - 1):  # Exclude the current state
        #     if not np.array_equal(ego_obs[i], self.padded_val):  # Check if the observation is not a padded value        
        for i in range(min(len(self.save_past_ego), self.render_histL)):
            val_hist_obs = self.save_past_ego[i]            
            if not np.array_equal(val_hist_obs, self.padded_val):
                hist_x, hist_y = val_hist_obs[:2]
                if self.normalize_xy:
                    hist_x = (hist_x + 1) * self.max_x / 2
                    hist_y = (hist_y + 1) * self.max_y / 2
                hist_screen_x = int(hist_x * scale_x)
                hist_screen_y = int(self.screen_height - hist_y * scale_y)  # Flip y-coordinate
                pygame.draw.circle(self.screen, (255, 0, 0), (hist_screen_x, hist_screen_y), 3)
        #Draw expert_obs
        if not np.array_equal(self.expert_obs[-1], self.padded_val):       
            for i in range(min(len(self.save_past_exp)-1, self.render_histL)):
                val_hist_obs = self.save_past_exp[i]            
                if not np.array_equal(val_hist_obs, self.padded_val):
                    hist_x, hist_y = val_hist_obs[:2]
                    if self.normalize_xy:
                        hist_x = (hist_x + 1) * self.max_x / 2
                        hist_y = (hist_y + 1) * self.max_y / 2
                    hist_screen_x = int(hist_x * scale_x)
                    hist_screen_y = int(self.screen_height - hist_y * scale_y)  # Flip y-coordinate
                    pygame.draw.circle(self.screen, (0, 255, 0), (hist_screen_x, hist_screen_y), 3)
        # Draw neighboring ships
        neighbor_obs = self.current_obs['neighbors']
        for neighbor in neighbor_obs:
            if not np.array_equal(neighbor[-1], self.padded_val):
                neighbor_x, neighbor_y = neighbor[-1][:2]
                neighbor_h = neighbor[-1][3]
                if len(neighbor) > 1:
                    neighbor_h = utils.cal_course(neighbor[-2][0], neighbor[-2][1], neighbor_x, neighbor_y, neighbor_h)
                if self.normalize_xy:
                    neighbor_x = (neighbor_x + 1) * self.max_x / 2
                    neighbor_y = (neighbor_y + 1) * self.max_y / 2
                screen_x = int(neighbor_x * scale_x)
                screen_y = int(self.screen_height - neighbor_y * scale_y)  # Flip y-coordinate
                # pygame.draw.circle(self.screen, (0, 0, 255), (screen_x, screen_y), 5)
                cartwidth = 25.0
                cartheight = 10.0
                l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
                pole_coords = []
                for coord in [(l, b), (l, t), (r, t), (r+(r-l)/4, (t+b)/2), (r, b)]:
                    coord = pygame.math.Vector2(coord).rotate_rad(-neighbor_h)  # Note the negative sign
                    coord = (coord[0] + screen_x, coord[1] + screen_y)
                    pole_coords.append(coord)
    
                ship_color = (0, 0, 255)  # Red for ego ship
                gfxdraw.aapolygon(self.screen, pole_coords, ship_color)
                gfxdraw.filled_polygon(self.screen, pole_coords, ship_color)
                #history
                for i in range(len(neighbor) - 1):  # Exclude the current state
                    if not np.array_equal(neighbor[i], self.padded_val):  # Check if the observation is not a padded value
                        hist_x, hist_y = neighbor[i][:2]                    
                        hist_screen_x = int(hist_x * scale_x)
                        hist_screen_y = int(self.screen_height - hist_y * scale_y)  # Flip y-coordinate
                        pygame.draw.circle(self.screen, ship_color, (hist_screen_x, hist_screen_y), 3)

        # Draw goal position
        goal_x, goal_y = self.goal_position
        screen_goal_x = int(goal_x * scale_x)
        screen_goal_y = int(self.screen_height - goal_y * scale_y)  # Flip y-coordinate
        # Calculate the position to center the image at the goal point
        star_rect = self.star_img.get_rect(center=(screen_goal_x, screen_goal_y))
        # Draw the star image
        self.screen.blit(self.star_img, star_rect)
        # pygame.draw.circle(self.screen, (0, 255, 0), (screen_goal_x, screen_goal_y), 5)

        # Draw step count and other information
        step_text = self.font.render(f"Ego id: {self.ego_pos}, Step: {self.current_step}, CPD: {self.CPD:.2f}, Dis-to-goal: {self.compute_dis_to_goal():.2f}, Curv: {self.current_curv:.2f}", 
                                     True, (0, 0, 0))
        self.screen.blit(step_text, (10, 10))
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )        

    def close(self):
        pygame.quit()
    
    def compute_dis_to_goal(self):
        goal_x, goal_y = self.goal_position 
        distance_to_goal = np.sqrt((self.current_state['x'] - goal_x)**2 + (self.current_state['y'] - goal_y)**2)
        return distance_to_goal
    def _check_goal_reached(self):
        # Implement goal-checking logic here        
        return self.compute_dis_to_goal() < self.goal_threshold  
    
    def reset(self, seed=None, options=None):
        """
        Resets the environment to start a new episode.
        
        EPISODE INITIALIZATION PROCESS:
        1. Selects which ship trajectory to use as the ego agent
        2. Sets the starting state from the first recorded position
        3. Sets the goal as the last recorded position  
        4. Identifies overlapping neighbor ships in the temporal window
        5. Pre-computes expert actions from the recorded trajectory
        
        Returns:
            observation: Initial observation dict with ego state, neighbors, and goal
            info: Dict containing goal_position and pre-computed expert actions
        """
        super().reset(seed=seed)
        
        # If options specify which ship to use as ego, update ego_pos
        # This allows external control over which trajectory becomes the episode
        if options is not None and options.get('ego_pos', None) != None:
            self.ego_pos = options['ego_pos']
        
        # Reset step counter and metrics for the new episode
        self.current_step = 0
        self.CPD = 0  # Closest Point Distance to nearest neighbor
        self.lastdheading = 0 
        
        # GOAL SETTING: Extract the final position from this ship's recorded trajectory
        # The episode goal is to navigate from start to this end position
        final_state = self.ship_trajectories[self.ego_pos][-1]
        self.goal_position = (final_state[0], final_state[1])  # (x, y) in meters
        ego_goal_x, ego_goal_y = self.goal_position
        
        # Normalize goal coordinates if using normalized observation space
        if self.normalize_xy:            
            ego_goal_x = (ego_goal_x / self.max_x) * 2 - 1  # Map [0, max_x] to [-1, 1]
            ego_goal_y = (ego_goal_y / self.max_y) * 2 - 1  # Map [0, max_y] to [-1, 1]
        
        # Initialize observation dictionary structure
        # 'ego': history of ego ship's states, 'neighbors': nearby ships, 'goal': target position
        self.current_obs = {'ego': None, 'neighbors': None, 'FoR_neigh':None, 'nearest_dis': None, 'goal': np.array([ego_goal_x, ego_goal_y])} 
        self.expert_obs = None
        
        # STARTING STATE: Extract the first position from this ship's recorded trajectory
        # This becomes the initial state of the episode
        initial_state = self.ship_trajectories[self.ego_pos][0]  # [x, y, speed, heading, (drift)]
        initial_time = self.ship_times[self.ego_pos][0]  # Unix timestamp in seconds
        
        # Initialize ego observation history with padding
        # History tracks past observations for temporal context
        self.current_obs['ego'] = np.array([self.padded_val]*(self.observation_history_length + 1))
        self.current_obs['ego'][-1] = initial_state  # Most recent observation is the starting state
        
        # Set the current state dictionary with all state variables
        self.current_state = {            
            'x': initial_state[0],          # X position in meters
            'y': initial_state[1],          # Y position in meters
            'speed': initial_state[2],      # Speed in m/s
            'heading': initial_state[3],    # Heading in radians
            'timestamp': initial_time       # Unix timestamp
        }
        
        # Initialize tracking variables for episode statistics
        self.total_neigh = 0  # Total neighbor count across all steps
        self.max_neigh = 0    # Maximum neighbors observed in any single step
        self.save_past_ego = deque()  # Deque to store ego's trajectory for rendering
        self.save_past_exp = deque()  # Deque to store expert trajectory for comparison
        
        # Get initial observation including neighboring ships at this timestamp
        # This populates self.current_obs['neighbors'] with nearby ships
        observation = self._get_observation()
        
        # EXPERT ACTION GENERATION: Extract actions from recorded trajectory
        # These represent the "expert" behavior - what the real ship actually did
        self.exp_acclerations = []
        self.exp_actions = self.gen_actions_from_data()  # Returns None if trajectory has invalid movements
        
        # Initialize metrics for tracking episode performance
        self.total_speed = 0
        self.min_speed = float('inf')
        self.max_speed = float('-inf')
        self.total_acceleration = 0
        self.min_acceleration = float('inf')
        self.max_acceleration = float('-inf')
        self.total_curvature = 0
        self.min_curvature = float('inf')
        self.max_curvature = float('-inf')
        self.nearmiss_count = 0
        self.current_curv = 0
        self.totalADE = 0
        self.totalASE = 0
        self.totalAAE = 0
        self.stop_step = 0
        self.count_cur_change = 0
        self.prv_cur = 0
        self.total_veh_in = 0
        self.total_drift = 0
        self.min_drift = float('inf')
        self.max_drift = float('-inf')        
        observation = observation.copy()
        if self.drop_neighbor:
            observation.pop("neighbors", None)
        if self.use_dis_fea is False:
            observation.pop('nearest_dis', None)
        if self.use_FoR is True:                
            observation['neighbors'] = observation['FoR_neigh']
        if self.use_time_fea:
            observation.update({'time': utils.encode_unix_time(initial_time)})
        observation.pop('FoR_neigh', None)        
        return observation, {'goal_position': self.goal_position, 'actions': self.exp_actions}
                
    def step(self, action):
        # Unpack the action
        dx, dy, dheading = action              
        # Scale the actions
        if self.normalize_xy:
            # Convert normalized dx, dy to actual distances
            dx_scaled = dx * (self.max_x / 2)
            dy_scaled = dy * (self.max_y / 2)
        else:
            if self.scale_act is False:
                dx_scaled = dx
                dy_scaled = dy
            else:
                # Scale dx, dy based on the environment size            
                dx_scaled = dx * (self.max_x / 20)  # Max step is 5% of environment width
                dy_scaled = dy * (self.max_y / 20)  # Max step is 5% of environment height

        # Scale dheading (same for both normalized and non-normalized cases)
        if self.scale_act:
            dheading_scaled = dheading * (np.pi / 2)  # Max heading change is 90 degrees
        else:
            dheading_scaled = dheading
        self.lastdheading = dheading_scaled
        self.save_past_ego.appendleft(self.current_obs["ego"][-1])
        # Update position and heading without clipping
        new_x = self.current_state['x'] + dx_scaled       
        new_y = self.current_state['y'] + dy_scaled
        new_heading = (self.current_state['heading'] + dheading_scaled) % (2 * np.pi)

        #Cal drift metrics:
        val_drift = utils.calculate_drift_two_points(self.current_state['x'], self.current_state['y'], new_x, new_y, new_heading)
        self.min_drift = min(self.min_drift, val_drift)
        self.max_drift = max(self.max_drift, val_drift)
        self.total_drift += abs(val_drift)
        
        # Calculate new speed (optional, depends on your requirements)
        new_speed = np.sqrt(dx_scaled**2 + dy_scaled**2)/self.second_perts  # Example: speed as magnitude of movement

        # Update timestamp
        new_timestamp = self.current_state['timestamp'] + self.second_perts  

        # Check if the new position is within the region of interest
        within_region = (0 <= new_x <= self.max_x) and (0 <= new_y <= self.max_y)

        # Store previous state for reward calculation
        previous_state = self.current_state.copy()

        # Update current state
        self.current_state = {
            'x': new_x,
            'y': new_y,
            'heading': new_heading,
            'speed': new_speed,
            'timestamp': new_timestamp
        }
         # Calculate acceleration
        old_speed = previous_state['speed']
        acceleration = (new_speed - old_speed) / self.second_perts

        # Calculate curvature (using the formula for curvature of a parametric curve)
        dx_dt = new_speed * np.cos(new_heading)
        dy_dt = new_speed * np.sin(new_heading)
        d2x_dt2 = acceleration * np.cos(new_heading) - new_speed * np.sin(new_heading) * dheading_scaled
        d2y_dt2 = acceleration * np.sin(new_heading) + new_speed * np.cos(new_heading) * dheading_scaled
        if (dx_dt**2 + dy_dt**2 < 1e-3): 
            curvature = 0   # If there's no movement at all, set curvature to zero
        else:
            #ref: https://mathworld.wolfram.com/Curvature.html#:~:text=)%20then%20gives-,(13),-For%20a%20two
            curvature = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2)**1.5
            # if curvature > 1:
            #     print(old_speed, previous_state["heading"])
            #     print(new_heading, new_speed)
            #     print((dx_dt**2 + dy_dt**2))
            #     assert 1==-1

        # Update metrics
        self.total_speed += new_speed
        self.min_speed = min(self.min_speed, new_speed)
        self.max_speed = max(self.max_speed, new_speed)
        self.total_acceleration += acceleration
        self.min_acceleration = min(self.min_acceleration, acceleration)
        self.max_acceleration = max(self.max_acceleration, acceleration)
        self.min_curvature = min(self.min_curvature, curvature)
        self.max_curvature = max(self.max_curvature, curvature)
        self.total_curvature += curvature
        if (curvature >= 0 and self.prv_cur < 0) or (curvature < 0 and self.prv_cur >= 0):
            self.count_cur_change += 1
        self.prv_cur = curvature
        #News
        if len(self.exp_actions) > self.current_step:
            exp_x = self.ship_trajectories[self.ego_pos][self.current_step+1][0]
            exp_y = self.ship_trajectories[self.ego_pos][self.current_step+1][1]
            self.totalADE += (new_x - exp_x)**2 + (new_y - exp_y)**2
            self.totalAAE += np.abs(acceleration - self.exp_acclerations[self.current_step])
            self.totalASE += np.abs(dheading_scaled - self.exp_actions[self.current_step][2]*(np.pi / 2))
            self.stop_step += 1   
                  

        if self.CPD < self.nearmiss_threshold: #555 meters mean nearmiss
            self.nearmiss_count += 1

        # Increment step counter
        self.current_step += 1

        # Determine if the episode is done or truncated
        done = self._check_goal_reached()  # Implement this method to check if the goal is reached
        dis_to_goal = self.compute_dis_to_goal()
        truncated = not within_region or self.current_step >= self.max_steps
        terminated = done 

        # Get new observation
        observation = self._get_observation()  #This function also change self.current_obs      

        # Calculate reward
        reward = 0
        if self.normalize_xy:
            # Calculate reward in normalized space
            norm_previous_x = (previous_state['x'] / self.max_x) * 2 - 1
            norm_previous_y = (previous_state['y'] / self.max_y) * 2 - 1
            norm_new_x = (new_x / self.max_x) * 2 - 1
            norm_new_y = (new_y / self.max_y) * 2 - 1
            # reward = -np.sqrt((norm_new_x - norm_previous_x)**2 + (norm_new_y - norm_previous_y)**2)
        else:
            # Calculate reward in original space
            pass
            # reward = -np.sqrt((new_x - previous_state['x'])**2 + (new_y - previous_state['y'])**2)

        # Apply a penalty if the agent moves out of bounds
        if truncated:
            reward = -100  # You can adjust this penalty as needed
        if done:
            reward = 1
            
        # Calculate additional metrics                       
        return_info = {'within_region': within_region, 'CPD': self.CPD, "dis_to_goal": dis_to_goal}
        #Reward features:
        goal_feature = -dis_to_goal
        if truncated:
            goal_feature = -10000
        if done:
            goal_feature = 1
        
        nearmiss_feature = 0
        if self.CPD < self.nearmiss_threshold: #555 meters mean nearmiss
            nearmiss_feature = -1
                
        acceleration_feature = -acceleration
        steer_feature = -new_speed*dheading_scaled
        return_info.update({
            'goal_feature': goal_feature,
            'nearmiss_feature': nearmiss_feature,
            'acceleration_feature': acceleration_feature,
            'steer_feature': steer_feature
        })        
        if terminated or truncated:
            avg_nbneigh = self.total_neigh / (self.current_step+1) 
            avg_speed = self.total_speed / self.current_step
            avg_acceleration = self.total_acceleration / self.current_step
            avg_drift = self.total_drift/self.current_step
            avg_curvature = self.total_curvature / self.current_step
            nearmiss_rate = self.nearmiss_count / self.current_step            
            self.current_curv = curvature
            curv_change_rate = self.count_cur_change/self.current_step
            #News:
            gc_ade = np.sqrt(self.totalADE)/self.stop_step
            mae_steer = self.totalASE/self.stop_step
            mae_accel = self.totalAAE/self.stop_step
            return_info.update({        
            'reach_goal': terminated,                   
            'avg_speed': avg_speed,
            'min_speed': self.min_speed,
            'max_speed': self.max_speed,
            'avg_acceleration': avg_acceleration,
            'min_acceleration': self.min_acceleration,
            'max_acceleration': self.max_acceleration,
            'avg_curvature': avg_curvature,
            'min_curvature': self.min_curvature,
            'max_curvature': self.max_curvature,
            'nearmiss_rate': nearmiss_rate*100, #in percentage
            "gc_ade": gc_ade,
            "mae_steer": mae_steer,
            "mae_accel": mae_accel,
            "curv_change_rate": curv_change_rate*100,
            "min_drift": self.min_drift,
            "max_drift": self.max_drift,
            "avg_drift": avg_drift,
            "eps_length": self.current_step,
            "avg_neigh": avg_nbneigh,
            "max_neigh": self.max_neigh
            })
        observation = observation.copy()        
        if self.drop_neighbor:
            observation.pop("neighbors", None)
        if self.use_dis_fea is False:
            observation.pop('nearest_dis', None)
        if self.use_FoR is True:                
            observation['neighbors'] = observation['FoR_neigh']
        if self.use_time_fea:
            observation.update({'time': utils.encode_unix_time(new_timestamp)})
        observation.pop('FoR_neigh', None)
        return observation, float(reward), terminated, truncated, return_info
    
    def _get_observation(self):
        """
        Constructs the observation for the current timestep.
        
        OBSERVATION CONSTRUCTION PROCESS:
        1. Update ego ship's observation history with current state
        2. Find all ships that overlap temporally with ego at current timestamp
        3. Calculate distances to all overlapping ships
        4. Select the N nearest neighbors
        5. Extract observation histories for each neighbor
        6. Pad if fewer than N neighbors exist
        
        Returns:
            dict containing:
                - 'ego': (history_len+1, features) array of ego's state history
                - 'neighbors': (N, history_len+1, features) array of neighbor histories
                - 'nearest_dis': (N,) array of distances to neighbors
                - 'FoR_neigh': neighbors transformed to ego's frame of reference (if enabled)
                - 'goal': (2,) array with goal position
        """
        # Get current timestamp from ego's state
        current_timestamp = self.current_state['timestamp']

        # UPDATE EGO OBSERVATION: Maintain sliding window history
        ego_obs = self.current_obs['ego'].copy()
        ego_obs = np.delete(ego_obs, 0, axis=0)  # Remove oldest observation (slide window)
        ego_obs = np.append(ego_obs, [self.padded_val], axis=0)  # Add placeholder for new observation
        
        # Populate current ego state from current_state dict
        ego_obs[-1][0] = self.current_state['x']       # X position
        ego_obs[-1][1] = self.current_state['y']       # Y position
        ego_obs[-1][2] = self.current_state['speed']   # Speed
        ego_obs[-1][3] = self.current_state['heading'] # Heading
        
        # Calculate drift angle if drift feature is enabled
        if self.use_drift_fea:            
            ego_obs[-1][4] = 0
            # Drift = angle between heading and actual movement direction
            if np.array_equal(self.current_obs['ego'][-1], self.padded_val) is False:
                past_x = self.current_obs['ego'][-1][0]
                past_y = self.current_obs['ego'][-1][1]
                ego_obs[-1][4] = utils.calculate_drift_two_points(past_x, past_y, ego_obs[-1][0], ego_obs[-1][1], ego_obs[-1][3])
        
        # Save updated ego observation
        self.current_obs['ego']= ego_obs
        
        # Get current ego position for distance calculations
        ego_x, ego_y = ego_obs[-1][:2]

        # FIND NEIGHBORING SHIPS: Query all ships that temporally overlap with ego
        distances = []
        for agent_id in self.overlap_idx[self.ego_pos]:  # overlap_idx contains ships in same time window
            if agent_id != self.ego_pos:  # Don't include ego as its own neighbor
                trajectory = self.ship_trajectories[agent_id]
                time = self.ship_times[agent_id]
                # Extract this neighbor's state at the current timestamp
                agent_obs = self._get_agent_observation(trajectory, time, current_timestamp)
                
                # Check if neighbor has valid data at this timestamp (not padded)
                if np.array_equal(agent_obs[-1], self.padded_val) is False:
                    agent_x, agent_y = agent_obs[-1][:2]
                    # Calculate Euclidean distance to ego ship
                    distance = np.sqrt((ego_x - agent_x)**2 + (ego_y - agent_y)**2)
                    distances.append((agent_id, distance))
        
        # Track neighbor statistics for metrics
        self.total_neigh += len(distances)  # Cumulative count across episode
        self.max_neigh = max(self.max_neigh, len(distances))  # Max in any single step
        
        # UPDATE EXPERT OBSERVATION: Track where ego "should" be according to recorded data
        # This is used for imitation learning comparison
        self.expert_obs = self._get_agent_observation(self.ship_trajectories[self.ego_pos], self.ship_times[self.ego_pos], current_timestamp)
        self.save_past_exp.appendleft(self.expert_obs[-1])
        
        # NEIGHBOR SELECTION: Sort by distance and select nearest N ships
        distances.sort(key=lambda x: x[1])
        nearest_neighbors = distances[:self.n_neighbor_agents]
        
        # Update CPD (Closest Point Distance) for collision risk metrics
        if (len(distances) != 0):
            self.CPD = distances[0][1]  # Distance to nearest neighbor
        else:
            self.CPD = 0  # No neighbors present
        
        # EXTRACT NEIGHBOR OBSERVATIONS: Get state histories for selected neighbors
        neighbor_obs = []
        nearest_dis_obs = []
        for agent_id, dis in nearest_neighbors:
            # Get observation history for this neighbor at current timestamp
            neighbor_obs.append(self._get_agent_observation(self.ship_trajectories[agent_id], 
                                                            self.ship_times[agent_id],
                                                            current_timestamp))
            nearest_dis_obs.append(dis)
        
        # PADDING: If fewer than N neighbors exist, pad with zeros
        while len(neighbor_obs) < self.n_neighbor_agents:
            neighbor_obs.append(self._get_padded_observation())
            nearest_dis_obs.append(np.sqrt(self.max_x**2+self.max_y**2))  # Max possible distance

        # Convert to numpy arrays for efficient processing
        self.current_obs['neighbors'] = np.array(neighbor_obs)
        
        # FRAME OF REFERENCE TRANSFORMATION: Convert neighbor positions to ego-centric coordinates
        if self.use_FoR:
            # Transform neighbor positions relative to ego's position and heading
            self.current_obs["FoR_neigh"] = utils.transform_neighbors_to_ego_frame(self.current_obs['ego'], self.current_obs['neighbors'])
        
        self.current_obs['nearest_dis'] = np.array(nearest_dis_obs, dtype=np.float32)
        
        return self.current_obs    

    def _get_agent_observation(self, trajectory, time, target_timestamp):
        """
        Extracts observation history for a ship at a specific timestamp.
        
        OBSERVATION HISTORY EXTRACTION:
        For a given ship and target timestamp:
        1. Define time window: [target_timestamp - history_length*timestep, target_timestamp]
        2. Find all recorded states within this window
        3. Pad beginning if ship wasn't yet active
        4. Extract state features (x, y, speed, heading, drift) for each timestep
        5. Normalize coordinates if required
        6. Pad end if not enough history exists
        
        This creates a temporal observation window showing where the ship has been.
        
        Args:
            trajectory: Array of recorded states for this ship
            time: Array of timestamps corresponding to trajectory states
            target_timestamp: Unix timestamp for which to extract observation
            
        Returns:
            numpy array of shape (history_length+1, features) containing state history
        """
        obs = []
        
        # Define the observation time window
        # Look back (history_length * time_per_step) seconds from target
        low_t = target_timestamp - self.observation_history_length*self.second_perts
        
        # Find all trajectory indices within the time window [low_t, target_timestamp]
        # Uses binary search on sorted time array for efficiency
        lst_ids = utils.find_indices_in_range_sorted(time, low_t, target_timestamp)
        
        if(len(lst_ids) != 0):
            # PAD BEGINNING: If ship's trajectory starts after low_t, pad with zeros
            # Calculate how many timesteps are missing at the start
            for i in np.arange(0, (time[lst_ids[0]] - low_t)/self.second_perts, dtype=int):
                obs.append(self.padded_val)  # Add padded observation for missing timesteps
            
            # EXTRACT STATES: Add recorded states within the time window
            for i in lst_ids:
                # Unpack state features from trajectory
                if self.use_drift_fea is False:
                    x, y, speed, heading = trajectory[i]
                else:
                    x, y, speed, heading, drift = trajectory[i]
                timestamp = time[i]
                
                # NORMALIZE COORDINATES: Convert to [-1, 1] range if enabled
                if self.normalize_xy:
                    x = (x / self.max_x) * 2 - 1  # Map [0, max_x] to [-1, 1]
                    y = (y / self.max_y) * 2 - 1  # Map [0, max_y] to [-1, 1]
                
                # Add state to observation history
                if self.use_drift_fea is False:
                    obs.append(np.array([x, y, speed, heading]))
                else:
                    obs.append(np.array([x, y, speed, heading, drift]))
        
        # PAD END: If observation history is shorter than required, pad with zeros
        # This happens when ship trajectory is shorter than history length
        while len(obs) < self.observation_history_length + 1:
            obs.append(self.padded_val)
        
        return np.array(obs)
    
    def _get_padded_observation(self):
        return [self.padded_val] * (self.observation_history_length + 1)
    
    def infer_action_from_trajectory(self, timestep):
        """
        Infers the action taken between two consecutive states in the recorded trajectory.
        
        ACTION INFERENCE PROCESS:
        Given state at timestep t and t+1 from recorded data:
        1. Calculate spatial displacement: dx = x[t+1] - x[t], dy = y[t+1] - y[t]
        2. Calculate heading change: dheading = heading[t+1] - heading[t] (shortest angle)
        3. Validate movements are within physically reasonable bounds
        4. Calculate acceleration from speed change
        5. Scale to action space range if needed
        
        Args:
            timestep: Index of the current state in the trajectory
            
        Returns:
            numpy array [dx, dy, dheading] if valid
            None if action exceeds bounds (trajectory has erratic movements)
        """
        # Get consecutive states from the recorded trajectory
        current_state = self.ship_trajectories[self.ego_pos][timestep] 
        next_state = self.ship_trajectories[self.ego_pos][timestep + 1]

        # Calculate position changes in meters
        dx = next_state[0] - current_state[0]  # Change in x position
        dy = next_state[1] - current_state[1]  # Change in y position
        
        # Calculate heading change, ensuring shortest angular path
        # (heading[t+1] - heading[t] + π) % 2π - π maps to [-π, π] range
        dheading = (next_state[3] - current_state[3] + np.pi) % (2 * np.pi) - np.pi
        
        # Track maximum displacements for validation
        self.max_dx = max(self.max_dx, abs(dx))
        if abs(dx) > 150:
            self.count_dx.append(abs(dx))  # Log unusually large displacements
        self.max_dy = max(self.max_dy, abs(dy))
        
        # Calculate acceleration for expert behavior tracking
        if timestep == 0:
            # For first step, calculate speed from displacement
            new_speed = np.sqrt(dx**2 + dy**2)/self.second_perts  # m/s
            # Acceleration = (new_speed - initial_speed) / time_interval
            self.exp_acclerations.append((new_speed - current_state[2])/self.second_perts)
        else:
            # For subsequent steps, calculate from previous speed
            new_speed = np.sqrt(dx**2 + dy**2)/self.second_perts
            prv_state = self.ship_trajectories[self.ego_pos][timestep-1]
            dx_old = current_state[0] - prv_state[0]
            dy_old = current_state[1] - prv_state[1]
            old_speed = np.sqrt(dx_old**2 + dy_old**2)/self.second_perts
            self.exp_acclerations.append((new_speed - old_speed)/self.second_perts)
        
        # VALIDATION: Check if movements are within acceptable bounds
        # These bounds ensure the trajectory represents normal ship behavior
        # Reject if: heading change > 90°, or displacement > 5% of environment size
        if abs(dheading) > np.pi/2 or abs(dx) > (self.max_x / 20) or abs(dy) >= (self.max_y / 20):
            return None  # Irregular/erratic action - reject this trajectory
        
        # SCALING: Transform actions to match the action space definition
        if self.normalize_xy:
            # For normalized observations, scale by half of max dimension
            dx = dx / (self.max_x / 2)
            dy = dy / (self.max_y / 2)        
        elif self.scale_act:
            # For scaled actions, normalize to [-1, 1] range
            dx = dx / (self.max_x / 20)   # Max displacement is 5% of width
            dy = dy / (self.max_y / 20)   # Max displacement is 5% of height
            dheading = dheading / (np.pi / 2)  # Max turn is 90°
        
        # Construct action array
        if self.scale_act is False:
            action = np.array([dx, dy, dheading])  # Unscaled (raw meter/radian values)
        else:
            # Clip to [-1, 1] to ensure actions are strictly within bounds
            action = np.clip([dx, dy, dheading], -1, 1)

        return action
    
    def gen_actions_from_data(self):
        """
        Extracts expert actions from the recorded trajectory of the ego ship.
        
        EXPERT ACTION EXTRACTION PROCESS:
        For each consecutive pair of states in the trajectory:
        1. Calculate positional change (dx, dy) between states
        2. Calculate heading change (dheading) between states
        3. Validate that changes are within acceptable bounds (not erratic)
        4. Scale actions to match the action space range
        
        Returns:
            numpy array of actions [(dx, dy, dheading), ...] if valid
            None if any action exceeds bounds (indicates erratic/invalid trajectory)
        """
        # Calculate trajectory length (number of transitions = states - 1)
        exp_traj_length = len(self.ship_trajectories[self.ego_pos]) - 1
        res_lst = []
        
        # Generate action for each transition in the recorded trajectory
        for i in range(exp_traj_length):
            # Extract action from state transition i -> i+1
            val_a = self.infer_action_from_trajectory(i)
            
            # If any action is invalid (too large movements), reject entire trajectory
            # This ensures only smooth, realistic trajectories are included in the dataset
            if val_a is None:
                return None
            
            res_lst.append(val_a)
        
        # Return as float32 array for compatibility with neural network training
        return np.array(res_lst, dtype=np.float32)

TIME_COL = "TIMESTAMP_UTC"
"""Convert geographic coordinates to local Cartesian coordinates"""
def lon_to_xpos(lon, origin_lon, origin_lat):
    return utils.haversine_distance(origin_lon, origin_lat, lon, origin_lat)    

def lat_to_ypos(lat, origin_lon, origin_lat):
    return utils.haversine_distance(origin_lon, origin_lat, origin_lon, lat)

def knots_to_ms(speed_knots):
    """Convert speed from 10xknots to meters per second"""
    return speed_knots * 0.0514444

def degrees_to_radians(degrees):
    """Convert angles from degrees to radians"""
    return np.radians(degrees)

def heading_to_2d_radians(heading_degrees):
    # Convert to radians
    heading_radians = np.radians(heading_degrees)
    
    # Adjust for coordinate system difference
    adjusted_radians = -1 * (heading_radians - np.pi/2)
    
    # Normalize to range [0, 2π)
    normalized_radians = adjusted_radians % (2 * np.pi)
    
    return normalized_radians

def radians_2d_to_heading(radians_2d):
    
    # Reverse the coordinate system adjustment
    heading_radians = -1 * radians_2d + np.pi/2
    
    # Convert to degrees
    heading_degrees = np.degrees(heading_radians)
    
    # Normalize to range [0, 360)
    normalized_degrees = heading_degrees % 360
    
    return normalized_degrees

def chunk_to_traj(chunk: pl.DataFrame, interpol_interval: timedelta, region_of_interest_array: np.ndarray):
    """
    Converts a single ship's raw AIS data chunk into a smoothed trajectory.
    
    TRAJECTORY GENERATION FROM RAW AIS DATA:
    Raw AIS data contains irregular, sparse position reports. This function:
    1. Extracts time, position, speed, heading from AIS records
    2. Converts geographic coordinates (lat/lon) to local Cartesian (x/y in meters)
    3. Converts speed from knots to m/s, heading from degrees to radians
    4. Defines regular time grid aligned to interpolation interval
    5. Interpolates positions, speed, heading onto regular time grid
    6. Validates that trajectory fits within region of interest
    
    This creates smooth, regularly-sampled trajectories suitable for RL training.
    
    Args:
        chunk: Polars DataFrame with 'all_records' column containing AIS position reports
        interpol_interval: Time between trajectory samples (e.g., 10 seconds)
        region_of_interest_array: [[lon_min, lat_min], [lon_max, lat_max]] bounds
        
    Returns:
        traj: numpy array of shape (n_timesteps, 4) with [x, y, speed, heading]
        t_np: numpy array of Unix timestamps corresponding to each state
        (None, None) if trajectory is too short or invalid
    """
    # Ensure interpolation interval is in integer seconds for alignment
    interpol_seconds = int(interpol_interval.total_seconds())
    if interpol_seconds != interpol_interval.total_seconds():
        raise ValueError("interpol_interval must be in integer seconds")

    # EXTRACT RAW DATA: Explode nested 'all_records' structure into flat DataFrame
    # Each row contains: timestamp, latitude, longitude, speed (10x knots), heading (degrees)
    df = chunk.select([
        pl.col('all_records').list.eval(pl.element().struct.field(TIME_COL)).alias('time'),
        pl.col('all_records').list.eval(pl.element().struct.field('LAT')).alias('lat'),
        pl.col('all_records').list.eval(pl.element().struct.field('LON')).alias('lon'),
        pl.col('all_records').list.eval(pl.element().struct.field('SPEED_KNOTSX10')).alias('speed'),
        pl.col('all_records').list.eval(pl.element().struct.field('HEADING')).alias('heading'),
    ]).explode(['time', 'lat', 'lon', 'speed', 'heading'])
    
    # DEFINE TIME GRID: Align start and end times to interpolation interval boundaries
    # start_time uses ceil: round up to next interval boundary
    start_time = df['time'].min().replace(microsecond=0)
    offset = (interpol_seconds - (start_time.second % interpol_seconds)) % interpol_seconds
    start_time = start_time + timedelta(seconds=offset)
    
    # end_time uses floor: round down to previous interval boundary
    end_time = df['time'].max().replace(microsecond=0)
    end_time = end_time - timedelta(seconds=end_time.second % interpol_seconds)

    # Validate time window is sufficient
    # After rounding to interpolation boundaries, we need at least one valid interval
    if start_time > end_time:
        # This happens when the chunk's time span is shorter than the interpolation interval
        # Example: chunk spans 12:34:52 to 12:34:57 (5 seconds)
        #   → start_time rounds to 12:35:00, end_time rounds to 12:34:50
        #   → start_time > end_time (invalid!)
        return None, None  # Trajectory too short for interpolation

    # Convert times to Unix timestamps (seconds since epoch) for numerical interpolation
    original_time = df['time'].map_elements(lambda x: int(x.timestamp()), return_dtype=pl.Int64)

    # DEFINE ORIGIN: Southwest corner of region becomes (0, 0) in local coordinates
    origin_lon, origin_lat = region_of_interest_array[0][0], region_of_interest_array[0][1]
    
    # COORDINATE CONVERSION: Geographic (lat/lon) -> Cartesian (x/y meters)
    original_time_np = original_time.to_numpy()
    # Convert latitudes to y positions using Haversine distance
    y_np = np.apply_along_axis(lambda x: lat_to_ypos(x, origin_lon, origin_lat), 0, df['lat'].to_numpy())
    # Convert longitudes to x positions using Haversine distance
    x_np = np.apply_along_axis(lambda x: lon_to_xpos(x, origin_lon, origin_lat), 0, df['lon'].to_numpy())
    # Convert speed from 10x knots to meters/second
    speed_np = np.apply_along_axis(knots_to_ms, 0, df['speed'].to_numpy())
    # Convert heading from compass degrees to 2D radians (math convention)
    heading_np = np.apply_along_axis(heading_to_2d_radians, 0, df['heading'].to_numpy())
    
    # CREATE REGULAR TIME GRID: Evenly spaced timestamps at interpolation interval
    t_np = np.arange(start_time.timestamp(), end_time.timestamp() + interpol_seconds, interpol_seconds, dtype=np.int64)
    
    # INTERPOLATION: Sample trajectory at regular intervals
    # Linear interpolation for x, y, speed (continuous quantities)
    traj_y = np.interp(t_np, original_time_np, y_np)
    traj_x = np.interp(t_np, original_time_np, x_np)
    traj_speed = np.interp(t_np, original_time_np, speed_np)
    # Angular interpolation for heading (handles wraparound at 0/2π)
    traj_heading = utils.shortest_path_angle_interp(t_np, original_time_np, heading_np)
    
    # ASSEMBLE TRAJECTORY: Stack into single array [x, y, speed, heading]
    traj = np.column_stack((traj_x, traj_y, traj_speed, traj_heading))
    traj = traj.astype(np.float32)  # Use float32 for memory efficiency
    
    return traj, t_np

def find_max_overlap_with_ego(df: pl.DataFrame, ego_id: int, n: int) -> pl.DataFrame:
    # Ensure ego_id exists in the DataFrame
    if ego_id not in df['SHIP_ID'].to_list():
        raise ValueError(f"ego_id {ego_id} not found in the DataFrame")

    # Get the ego ship's interval
    ego_row = df.filter(pl.col('SHIP_ID') == ego_id)
    ego_start = ego_row['start_time'][0]
    ego_end = ego_row['end_time'][0]

    # Calculate overlap for each row
    df_with_overlap = df.with_columns(
        overlap = (
            pl.when(pl.col('end_time') <= ego_start)
            .then(0)
            .when(pl.col('start_time') >= ego_end)
            .then(0)
            .otherwise(
                pl.when(pl.col('start_time') <= ego_start)
                .then(
                    pl.when(pl.col('end_time') <= ego_end)
                    .then(pl.col('end_time') - ego_start)
                    .otherwise(ego_end - ego_start)
                )
                .otherwise(
                    pl.when(pl.col('end_time') <= ego_end)
                    .then(pl.col('end_time') - pl.col('start_time'))
                    .otherwise(ego_end - pl.col('start_time'))
                )
            )
        )
    )

    # Sort by overlap (descending) and SHIP_ID (to ensure ego_id is first if tied)
    df_sorted = df_with_overlap.sort(['overlap', 'SHIP_ID'], descending=[True, False])

    # Get top n rows
    result = df_sorted.head(n)

    # Ensure ego_id is in the result
    if ego_id not in result['SHIP_ID'].to_list():
        result = pl.concat([
            ego_row,
            result.filter(pl.col('SHIP_ID') != ego_id).head(n - 1)
        ])
    elif result['SHIP_ID'][0] != ego_id:
        ego_index = result['SHIP_ID'].to_list().index(ego_id)
        result = pl.concat([
            result.slice(1, 1),  # New first row (previously top overlap)
            result.slice(0, 1),  # New second row (ego_id)
            result.slice(2, ego_index - 1) if ego_index > 1 else None,  # Rows between if any
            result.slice(ego_index + 1, None)  # Remaining rows
        ])

    return result

# Function to read static data (vessel information)
def read_static_data(file_pattern):
    return (pl.scan_csv(file_pattern, 
                        separator=";", 
                        null_values=["","NULL"])
            .select(["SHIP_ID", "VESSEL_TYPE"])
            .unique(subset=["SHIP_ID"])  # Remove duplicates based on SHIP_ID
            .collect())

# Function to read position data
def read_position_data(file_pattern):
    return (pl.scan_csv(file_pattern, 
                        separator=";", 
                        null_values=["","NULL"], 
                        try_parse_dates=True)
            .unique(subset=["SHIP_ID", "TIMESTAMP_UTC"])  # Remove duplicates based on SHIP_ID and TIMESTAMP_UTC
            .collect())


def create_minari_dataset(env, dataset_name, num_ships):
    """
    Creates a Minari dataset containing ship navigation episodes extracted from real AIS trajectory data.
    
    EPISODE GENERATION PROCESS:
    Each episode represents one ship's complete trajectory from start to goal position.
    The ego ship follows its expert trajectory from the data while observing neighboring ships.
    
    Episodes are recorded until termination/truncation, and actions are automatically
    truncated to match the actual episode length (no pre-validation needed).
    
    Args:
        env: The ship environment containing all preprocessed trajectories
        dataset_name: Name for the Minari dataset to be created
        num_ships: Total number of ship trajectories available in the environment
    """
    
    # PRE-FILTER: Check which trajectories will produce useful episodes
    # Skip trajectories that are too short or will terminate immediately
    print("Pre-filtering trajectories...")
    min_episode_length = 5  # Minimum number of steps for useful learning data
    valid_episodes = []
    
    # Get the base environment (unwrapped if it's wrapped)
    base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    
    for id_ego in range(num_ships):
        # Do a dry run to check episode viability (without DataCollector)
        obs, info = base_env.reset(seed=42, options={'ego_pos': id_ego})
        actions = info['actions']
        
        if actions is None:
            print(f"Episode {id_ego}: SKIP (invalid actions)")
            continue
        
        # Simulate to see how many steps it actually takes
        steps_taken = 0
        for i in range(len(actions)):
            action = actions[i]
            observation, reward, terminated, truncated, info = base_env.step(action)
            steps_taken += 1
            if terminated or truncated:
                break
        
        # Only include episodes with sufficient length
        if steps_taken >= min_episode_length:
            valid_episodes.append((id_ego, steps_taken, len(actions)))
            print(f"Episode {id_ego}: VALID ({steps_taken}/{len(actions)} steps)")
        else:
            print(f"Episode {id_ego}: SKIP (only {steps_taken} steps, min={min_episode_length})")
    
    print(f"\nFound {len(valid_episodes)} valid episodes out of {num_ships}")
    print(f"{'='*60}\n")
    
    # Wrap environment with DataCollector NOW, after filtering
    # Pass the original environment (not unwrapped) so env_spec is preserved
    env = minari.DataCollector(env, action_space=base_env.action_space, observation_space=base_env.observation_space)
    
    # Collect only the validated episodes
    episodes_collected = 0
    for id_ego, expected_steps, total_actions in valid_episodes:   
        print(f"Collecting episode {id_ego} ({expected_steps}/{total_actions} steps)...", end=" ")
        
        # Reset and collect this validated episode
        obs, info = env.reset(seed=42, options={'ego_pos': id_ego})
        actions = info['actions']
        
        # Execute the episode (we know it will complete successfully from pre-check)
        for i in range(len(actions)):        
            action = actions[i]
            observation, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        print("RECORDED")
        episodes_collected += 1
    
    print(f"\n{'='*60}")
    print(f"Dataset collection complete!")
    print(f"Episodes recorded: {episodes_collected}")
    print(f"Episodes skipped: {num_ships - episodes_collected} (invalid or too short)")
    print(f"Total processed: {num_ships}")
    print(f"{'='*60}\n")
    
    # Create and save the final Minari dataset with all collected episodes
    dataset = env.create_dataset(dataset_id=dataset_name,                                                            
                                author="jjvandenbroek",
                                author_email="jasperjvandenbroek@gmail.com")

