import gymnasium as gym
import pygame
import time
import numpy as np
from stable_baselines3 import PPO

# --- Configuration ---
MODEL_PATH = "models/ppo_cartpole"  # Path to your trained PPO model
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CART_COLOR = (0, 0, 150)  # Dark Blue
POLE_COLOR = (150, 0, 0)  # Dark Red
ARROW_COLOR = (0, 150, 0)  # Dark Green
BACKGROUND_COLOR = (255, 255, 255)  # White
RAIL_COLOR = (0, 0, 0)  # Black
FPS = 30  # Frames per second (controls simulation speed)
SCALE = 100  # Pixel-to-meter scaling factor
SLOW_MOTION_FACTOR = 3  # Factor to slow down the simulation
BACKGROUND_SPEED = 2  # Constant speed for background objects

# --- Background Objects ---
class BackgroundObject:
    def __init__(self, x, y, width, height, color, speed_factor):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.speed_factor = speed_factor  # Now used for relative initial position

    def move(self):
        # Move the object to the left at a constant speed
        self.rect.x -= BACKGROUND_SPEED

        # Reset position if it goes off-screen to the left
        if self.rect.right < 0:
            self.rect.left = SCREEN_WIDTH + self.rect.width * self.speed_factor

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)

# --- Pygame Initialization ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Inverted Pendulum Simulation")
clock = pygame.time.Clock()

# --- Load Environment and Model ---
env = gym.make("CartPole-v1")
model = PPO.load(MODEL_PATH)

# --- Create Background Objects ---
background_objects = [
    BackgroundObject(100, 450, 50, 80, (150, 75, 0), 0.2),  # Example building
    BackgroundObject(400, 480, 30, 50, (150, 75, 0), 0.5),  # Another building
    BackgroundObject(700, 470, 40, 60, (150, 75, 0), 0.8),  # Yet another building
    BackgroundObject(250, 520, 20, 30, (0, 150, 0), 0.3),  # Example tree
    BackgroundObject(550, 510, 25, 40, (0, 150, 0), 0.6),  # Another tree
]

# --- Helper Functions ---
def draw_cart(x, theta, action):
    """Draws the cart, pole, force arrow, and rail on the screen."""
    cart_x = x * SCALE + SCREEN_WIDTH // 2  # Scale and center the cart
    cart_y = SCREEN_HEIGHT * 4 // 5  # Position cart near the bottom
    cart_width = 80
    cart_height = 40

    # Cart rectangle
    cart_rect = pygame.Rect(
        cart_x - cart_width // 2, cart_y - cart_height // 2, cart_width, cart_height
    )
    pygame.draw.rect(screen, CART_COLOR, cart_rect, 0, 5)  # Filled with rounded corners
    pygame.draw.rect(screen, (0, 0, 0), cart_rect, 2, 5)  # Black outline

    # Pole
    pole_length = 100
    pole_end_x = cart_x + pole_length * np.sin(theta)
    pole_end_y = cart_y - pole_length * np.cos(theta)
    pygame.draw.line(screen, POLE_COLOR, (cart_x, cart_y), (pole_end_x, pole_end_y), 10)

    # Force Arrow
    arrow_length = 50
    arrow_head_size = 10
    if action == 1:
        start_pos = (cart_x, cart_y - cart_height // 2)
        end_pos = (cart_x + arrow_length, cart_y - cart_height // 2)
    else:
        start_pos = (cart_x, cart_y - cart_height // 2)
        end_pos = (cart_x - arrow_length, cart_y - cart_height // 2)

    pygame.draw.line(screen, ARROW_COLOR, start_pos, end_pos, 5)
    angle = np.arctan2(start_pos[1] - end_pos[1], start_pos[0] - end_pos[0])
    pygame.draw.polygon(
        screen,
        ARROW_COLOR,
        [
            end_pos,
            (
                end_pos[0] + arrow_head_size * np.cos(angle - np.pi / 6),
                end_pos[1] + arrow_head_size * np.sin(angle - np.pi / 6),
            ),
            (
                end_pos[0] + arrow_head_size * np.cos(angle + np.pi / 6),
                end_pos[1] + arrow_head_size * np.sin(angle + np.pi / 6),
            ),
        ],
    )

    # Rail
    rail_y = cart_y + cart_height // 2 + 5
    pygame.draw.line(screen, RAIL_COLOR, (0, rail_y), (SCREEN_WIDTH, rail_y), 4)

    # Wheel
    wheel_radius = 10
    wheel_y = cart_y + cart_height // 2 + wheel_radius // 2 - 5
    pygame.draw.circle(screen, (0, 0, 0), (cart_x - cart_width // 4, wheel_y), wheel_radius)
    pygame.draw.circle(screen, (0, 0, 0), (cart_x + cart_width // 4, wheel_y), wheel_radius)

# --- Main Simulation Loop ---
def simulate():
    obs, info = env.reset()
    done = False
    total_reward = 0
    sim_step = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        if sim_step % SLOW_MOTION_FACTOR == 0:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Drawing ---
        screen.fill(BACKGROUND_COLOR)

        # Move and draw background objects (now independent of cart speed)
        for obj in background_objects:
            obj.move()
            obj.draw(screen)

        draw_cart(obs[0], obs[2], action)  # Draw cart, pole, force arrow, and rail
        pygame.display.flip()

        clock.tick(FPS)
        sim_step += 1

    print(f"Total Reward: {total_reward}")

# --- Run the simulation ---
if __name__ == "__main__":
    simulate()
    env.close()
    pygame.quit()