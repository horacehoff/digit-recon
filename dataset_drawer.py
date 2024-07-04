import random
import pygame
import os

pygame.init()

width, height = 500, 500
grid_size = 10
pixel_size = width // grid_size
white = (255, 255, 255)
black = (0, 0, 0)
number_folder = "nine"


screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("PIXELATED DIGITS -- DATASET CREATOR 3000")

grid = [[white for _ in range(grid_size)] for _ in range(grid_size)]


def draw_grid():
    for row in range(grid_size):
        for col in range(grid_size):
            pygame.draw.rect(screen, grid[row][col],
                             (col * pixel_size, row * pixel_size, pixel_size, pixel_size))
            pygame.draw.rect(screen, black,
                             (col * pixel_size, row * pixel_size, pixel_size, pixel_size), 1)


def save_image():
    image_surface = pygame.Surface((grid_size, grid_size))
    for row in range(grid_size):
        for col in range(grid_size):
            image_surface.set_at((col, row), grid[row][col])

    file_path = os.path.join("data/training/" + number_folder, number_folder + str(random.randrange(0, 100000)) + ".png")
    pygame.image.save(image_surface, file_path)
    print(f"Image saved to {file_path}")
    reset_grid()


def reset_grid():
    global grid
    grid = [[white for _ in range(grid_size)] for _ in range(grid_size)]

running = True
drawing = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                save_image()
            elif event.key == pygame.K_r:
                reset_grid()

    if drawing:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        col = mouse_x // pixel_size
        row = mouse_y // pixel_size
        if 0 <= col < grid_size and 0 <= row < grid_size:
            grid[row][col] = black

    screen.fill(white)
    draw_grid()
    pygame.display.flip()

pygame.quit()