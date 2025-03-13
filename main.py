import gym
from gym import spaces
import numpy as np
import pygame
import math
import sys
import argparse

# ------------------------------
# 1. Define the Custom Gym Environment
# ------------------------------

class DroneFireExtEnv(gym.Env):
    """
    A custom 2D environment where a drone must hover and navigate toward a fire.
    The drone is controlled by two continuous thrust inputs (left and right),
    and the goal is to reach and 'extinguish' the fire.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DroneFireExtEnv, self).__init__()
        # Time step and physics constants
        self.dt = 0.05              # Reduced from 0.1 to 0.05 for slower simulation
        self.gravity = 9.8         # gravitational acceleration
        self.thrust_scale = 20.0   # scaling for the combined thrust
        self.torque_scale = 5.0    # scaling for angular acceleration

        # Define environment boundaries (for state clipping and reward)
        self.x_bounds = (-100, 100)
        self.y_bounds = (0, 200)

        # For rendering via Pygame
        self.screen_width = 600
        self.screen_height = 400

        # Load the drone sprite (initialize to None, load once during render)
        self.drone_sprite = None
        self.previous_distance = None

        # Define the observation space.
        # State consists of:
        # [drone_x, drone_y, drone_vx, drone_vy, drone_angle, drone_angular_vel, fire_dx, fire_dy]
        low_obs = np.array([self.x_bounds[0], self.y_bounds[0],
                            -np.finfo(np.float32).max, -np.finfo(np.float32).max,
                            -np.pi, -np.finfo(np.float32).max,
                            -np.finfo(np.float32).max, -np.finfo(np.float32).max],
                           dtype=np.float32)
        high_obs = np.array([self.x_bounds[1], self.y_bounds[1],
                             np.finfo(np.float32).max, np.finfo(np.float32).max,
                             np.pi, np.finfo(np.float32).max,
                             np.finfo(np.float32).max, np.finfo(np.float32).max],
                            dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # Define action space: two continuous thrust values in [0, 1] for left and right motors.
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # Initialize Pygame variables when render is first called.
        self.screen = None
        self.clock = None

        # Initialize environment state.
        self.reset()

    def reset(self, seed=None, options=None):
        # Handle seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Reset drone state
        self.drone_pos = np.array([0.0, 10.0])
        self.drone_vel = np.array([0.0, 0.0])
        self.drone_angle = 0.0
        self.drone_angular_vel = 0.0
        self.time = 0.0
        
        # Reset fire position randomly
        self.fire_pos = np.array([
            np.random.uniform(self.x_bounds[0]/2, self.x_bounds[1]/2),
            np.random.uniform(self.y_bounds[0] + 50, self.y_bounds[1] - 50)
        ])
        
        # Initialize previous_distance for progress reward
        self.previous_distance = np.sqrt(np.sum((self.drone_pos - self.fire_pos)**2))
        
        # Return both observation and an empty info dict
        return self._get_obs(), {}

    def step(self, action):
        # Ensure action is within bounds.
        action = np.clip(action, self.action_space.low, self.action_space.high)
        thrust_left, thrust_right = action

        # Calculate total thrust and use it to compute accelerations.
        total_thrust = (thrust_left + thrust_right) * self.thrust_scale

        # Horizontal acceleration approximated via tilt (sin component).
        ax = math.sin(self.drone_angle) * total_thrust

        # Vertical acceleration: thrust minus gravity.
        ay = total_thrust - self.gravity

        # Update velocities using simple Euler integration.
        self.drone_vel[0] += ax * self.dt
        self.drone_vel[1] += ay * self.dt

        # Update positions.
        self.drone_pos += self.drone_vel * self.dt

        # Update angular velocity and angle.
        angular_acc = (thrust_left - thrust_right) * self.torque_scale
        self.drone_angular_vel += angular_acc * self.dt
        self.drone_angle += self.drone_angular_vel * self.dt
        # Keep angle within [-pi, pi].
        self.drone_angle = (self.drone_angle + np.pi) % (2 * np.pi) - np.pi

        self.time += self.dt

        # Calculate rewards
        distance = np.sqrt(np.sum((self.drone_pos - self.fire_pos)**2))
        
        # Stability rewards/penalties
        angular_velocity_penalty = -0.5 * abs(self.drone_angular_vel)  # Penalize fast rotation
        tilt_penalty = -1.0 * abs(self.drone_angle)  # Penalize extreme tilting
        movement_penalty = -0.3 * (abs(self.drone_vel[0]) + abs(self.drone_vel[1]))  # Penalize fast movement
        
        # Navigation reward
        distance_reward = -0.1 * distance  # Small continuous reward for being closer
        
        # Progress reward (compare to previous distance)
        progress_reward = 0.5 * (self.previous_distance - distance)
        self.previous_distance = distance
        
        # Initialize done flag
        done = False
        
        # Success reward
        success_reward = 0.0
        if distance < 5.0:
            success_reward = 100.0
            done = True
        
        # Crash penalty
        crash_penalty = 0.0
        if not (self.y_bounds[0] <= self.drone_pos[1] <= self.y_bounds[1]):
            crash_penalty = -50.0
            done = True
        
        # Combine all rewards
        reward = (
            angular_velocity_penalty +
            tilt_penalty +
            movement_penalty +
            distance_reward +
            progress_reward +
            success_reward +
            crash_penalty
        )

        info = {
            'distance_to_fire': distance,
            'time': self.time,
            'stability_score': angular_velocity_penalty + tilt_penalty
        }

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        # Observation: [drone_x, drone_y, drone_vx, drone_vy, drone_angle, drone_angular_vel, fire_dx, fire_dy]
        fire_rel = self.fire_pos - self.drone_pos
        obs = np.array([self.drone_pos[0], self.drone_pos[1],
                        self.drone_vel[0], self.drone_vel[1],
                        self.drone_angle, self.drone_angular_vel,
                        fire_rel[0], fire_rel[1]], dtype=np.float32)
        return obs

    def render(self, mode='human'):
        # Initialize Pygame display on first call.
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("2D Drone Fire Extinguishing Simulation")
            self.clock = pygame.time.Clock()

        # Clear the screen.
        self.screen.fill((200, 200, 200))  # light grey background

        # Helper: convert environment coordinates to screen coordinates.
        def env_to_screen(pos):
            # Map x from self.x_bounds to [0, screen_width]
            x = int((pos[0] - self.x_bounds[0]) / (self.x_bounds[1] - self.x_bounds[0]) * self.screen_width)
            # Map y from self.y_bounds to [screen_height, 0] (flip vertical axis)
            y = int(self.screen_height - (pos[1] - self.y_bounds[0]) / (self.y_bounds[1] - self.y_bounds[0]) * self.screen_height)
            return (x, y)

        # Draw the fire as a red circle.
        fire_screen = env_to_screen(self.fire_pos)
        pygame.draw.circle(self.screen, (255, 0, 0), fire_screen, 10)

        # Draw the drone using the external sprite.
        drone_screen = env_to_screen(self.drone_pos)
        # Load the sprite if it hasn't been loaded yet.
        if self.drone_sprite is None:
            try:
                self.drone_sprite = pygame.image.load("/Users/james/Desktop/RL-Drone-Flying/Drone GIF Drawing.gif").convert_alpha()
                # Optionally scale the sprite to a desired size.
                self.drone_sprite = pygame.transform.scale(self.drone_sprite, (40, 40))
            except Exception as e:
                print("Error loading drone sprite:", e)
                # Fallback to a simple rectangle if image loading fails.
                self.drone_sprite = pygame.Surface((40, 40))
                self.drone_sprite.fill((0, 0, 255))
        # Rotate the sprite based on the drone's angle.
        rotated_sprite = pygame.transform.rotate(self.drone_sprite, -math.degrees(self.drone_angle))
        rect = rotated_sprite.get_rect(center=drone_screen)
        self.screen.blit(rotated_sprite, rect)

        # Optionally, display simulation info.
        font = pygame.font.SysFont("Arial", 14)
        text_surface = font.render(f"Time: {self.time:.1f} s", True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(30)  # Aim for 30 frames per second

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

# ------------------------------
# 2. RL Training Code (Using Stable Baselines3 PPO)
# ------------------------------

def train_model():
    from stable_baselines3 import PPO
    
    # Create the environment
    env = DroneFireExtEnv()
    
    # Initialize the PPO agent with parameters favoring stable behavior
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0001,  # Smaller learning rate for more stable learning
        n_steps=2048,  # Longer horizon to better understand consequences
        batch_size=64,
        n_epochs=10,
        gamma=0.99,  # High gamma to value long-term stability
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )
    
    # Train for longer to ensure good behavior
    model.learn(total_timesteps=500000)
    model.save("drone_fire_model_stable")

# ------------------------------
# 3. Simulation Code Using the Trained Model
# ------------------------------

def simulate_trained_model(model_path="drone_fire_model.zip"):
    from stable_baselines3 import PPO
    
    try:
        # Initialize Pygame BEFORE creating the environment
        pygame.init()
        if not pygame.display.get_init():
            pygame.display.init()
            
        env = DroneFireExtEnv()
        # Load the trained model
        model = PPO.load(model_path)
        obs = env.reset()
        done = False
        
        while not done:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break
            if done:
                break
                
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            
    except Exception as e:
        print(f"Simulation error: {e}")
    finally:
        pygame.quit()
        if env:
            env.close()

def human_control_mode():
    pygame.init()
    env = DroneFireExtEnv()
    obs = env.reset()
    done = False
    
    # Initialize control variables
    left_thrust = 0.5
    right_thrust = 0.5
    thrust_delta = 0.05  # Reduced for finer control
    
    print("\nHuman Control Mode Instructions:")
    print("↑ (Up Arrow) - Increase thrust")
    print("↓ (Down Arrow) - Decrease thrust")
    print("← (Left Arrow) - Rotate left")
    print("→ (Right Arrow) - Rotate right")
    print("SPACE - Stabilize (reset thrusters to balanced)")
    print("R - Reset environment")
    print("ESC - Quit\n")
    
    try:
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True
                    elif event.key == pygame.K_r:
                        obs = env.reset()
                        left_thrust = 0.5
                        right_thrust = 0.5
                    elif event.key == pygame.K_SPACE:
                        # Quick stabilize
                        left_thrust = 0.5
                        right_thrust = 0.5
            
            # Continuous key state handling
            keys = pygame.key.get_pressed()
            
            # Simplified thrust controls
            if keys[pygame.K_UP]:  # Increase both thrusters
                left_thrust = min(1.0, left_thrust + thrust_delta)
                right_thrust = min(1.0, right_thrust + thrust_delta)
            if keys[pygame.K_DOWN]:  # Decrease both thrusters
                left_thrust = max(0.0, left_thrust - thrust_delta)
                right_thrust = max(0.0, right_thrust - thrust_delta)
            if keys[pygame.K_LEFT]:  # Rotate left
                left_thrust = max(0.0, left_thrust - thrust_delta)
                right_thrust = min(1.0, right_thrust + thrust_delta)
            if keys[pygame.K_RIGHT]:  # Rotate right
                left_thrust = min(1.0, left_thrust + thrust_delta)
                right_thrust = max(0.0, right_thrust - thrust_delta)
            
            action = np.array([left_thrust, right_thrust])
            obs, reward, done, info = env.step(action)
            
            print(f"\rReward: {reward:+.2f} | Distance: {info['distance_to_fire']:.2f} | Time: {info['time']:.1f}s | Thrust L/R: {left_thrust:.2f}/{right_thrust:.2f}", end="")
            
            env.render()
            pygame.time.wait(50)
            
    except Exception as e:
        print(f"\nError in human control mode: {e}")
    finally:
        env.close()
        pygame.quit()

def record_human_demonstration():
    pygame.init()
    env = DroneFireExtEnv()
    demonstrations = []  # Store (state, action, reward) tuples
    
    obs = env.reset()
    done = False
    
    # Initialize control variables
    left_thrust = 0.5  # Start with balanced thrust
    right_thrust = 0.5
    thrust_delta = 0.1  # How much to change thrust per keypress
    
    print("\nRecording Human Demonstration Mode")
    print("Instructions:")
    print("W/S - Increase/Decrease both thrusters")
    print("A/D - Increase left/right thruster")
    print("Q/E - Decrease left/right thruster")
    print("R - Reset environment")
    print("ESC - Quit and save demonstration\n")
    
    try:
        while not done:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True
                    elif event.key == pygame.K_r:
                        obs = env.reset()
                        left_thrust = 0.5
                        right_thrust = 0.5
            
            # Continuous key state handling
            keys = pygame.key.get_pressed()
            
            # Thrust controls
            if keys[pygame.K_w]:  # Increase both thrusters
                left_thrust = min(1.0, left_thrust + thrust_delta)
                right_thrust = min(1.0, right_thrust + thrust_delta)
            if keys[pygame.K_s]:  # Decrease both thrusters
                left_thrust = max(0.0, left_thrust - thrust_delta)
                right_thrust = max(0.0, right_thrust - thrust_delta)
            if keys[pygame.K_a]:  # Increase left thruster
                left_thrust = min(1.0, left_thrust + thrust_delta)
            if keys[pygame.K_d]:  # Increase right thruster
                right_thrust = min(1.0, right_thrust + thrust_delta)
            if keys[pygame.K_q]:  # Decrease left thruster
                left_thrust = max(0.0, left_thrust - thrust_delta)
            if keys[pygame.K_e]:  # Decrease right thruster
                right_thrust = max(0.0, right_thrust - thrust_delta)
            
            # Take action and get feedback
            action = np.array([left_thrust, right_thrust])
            new_obs, reward, done, info = env.step(action)
            
            # Record the demonstration step
            demonstrations.append({
                'observation': obs,
                'action': action,
                'reward': reward,
                'next_observation': new_obs,
                'done': done
            })
            
            obs = new_obs
            
            # Display current state information
            print(f"\rRecording: Steps={len(demonstrations)} | Reward: {reward:+.2f} | Distance to Fire: {info['distance_to_fire']:.2f} | Time: {info['time']:.1f}s", end="")
            
            env.render()
            pygame.time.wait(50)  # Slower update rate for better control
            
    except Exception as e:
        print(f"\nError in recording mode: {e}")
    finally:
        if len(demonstrations) > 0:
            print(f"\nSaving demonstration with {len(demonstrations)} steps...")
            np.save('human_demo.npy', demonstrations)
            print("Demonstration saved as 'human_demo.npy'")
        env.close()
        pygame.quit()
    
    return demonstrations

def train_from_demonstrations(demo_path='human_demo.npy'):
    from stable_baselines3 import PPO
    
    try:
        # Load demonstrations
        print("Loading demonstrations...")
        demonstrations = np.load(demo_path, allow_pickle=True)
        print(f"Loaded {len(demonstrations)} demonstration steps")
        
        # Create environment and model
        env = DroneFireExtEnv()
        model = PPO("MlpPolicy", env, verbose=1)
        
        # Initialize replay buffer with demonstrations
        print("Initializing model with demonstrations...")
        for demo in demonstrations:
            # Pre-training step
            model.learn(total_timesteps=1, reset_num_timesteps=False)
            
            # Add demonstration to replay buffer
            model.replay_buffer.add(
                obs=demo['observation'],
                action=demo['action'],
                reward=demo['reward'],
                next_obs=demo['next_observation'],
                done=demo['done']
            )
        
        print("Starting additional training...")
        # Continue training with RL
        model.learn(total_timesteps=100000)
        
        # Save the trained model
        model.save("drone_fire_model_with_demos")
        print("Model saved as 'drone_fire_model_with_demos.zip'")
        
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        env.close()

# ------------------------------
# 4. Main Entry Point: Choose Training or Simulation Mode
# ------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="2D Drone Fire Extinguishing RL Project")
    parser.add_argument("--train", action="store_true", help="Train the RL model")
    parser.add_argument("--simulate", action="store_true", help="Run simulation with the trained model")
    parser.add_argument("--human", action="store_true", help="Run in human control mode")
    parser.add_argument("--record", action="store_true", help="Record human demonstrations")
    parser.add_argument("--train-from-demos", action="store_true", help="Train using recorded demonstrations")
    args = parser.parse_args()

    if args.record:
        record_human_demonstration()
    elif args.train_from_demos:
        train_from_demonstrations()
    elif args.train:
        train_model()
    elif args.simulate:
        simulate_trained_model()
    elif args.human:
        human_control_mode()
    else:
        print("Please specify --train to train the model, --simulate to run the simulation, or --human for human control mode.")
