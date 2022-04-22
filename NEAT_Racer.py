import sys
import os
import random
import math
import neat
import pygame
from pygame.locals import *
import cProfile
import pstats
 
pygame.init()
pygame.font.init()


WIDTH, HEIGHT = 800, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))

GRAY = (128, 128, 128)

pygame.display.set_icon(pygame.image.load("icon.png"))
pygame.display.set_caption("NEAT Racer")

CAR_SIZE = (70,150)
CAR_IMG = pygame.image.load("car.png")
CAR_IMG = pygame.transform.smoothscale(CAR_IMG, CAR_SIZE)

OBSTACLE_SIZE = (100,60)
OBSTACLE_IMG = pygame.image.load("obstacle.png")
OBSTACLE_IMG = pygame.transform.smoothscale(OBSTACLE_IMG, OBSTACLE_SIZE)

ROAD_WIDTH = 460
ROAD_HEIGHT = HEIGHT
ROAD = Rect(0,0,ROAD_WIDTH,ROAD_HEIGHT)

GEN = 0


class Car:
    MAX_X = ROAD_WIDTH - CAR_SIZE[0]
    SPEED = 8

    def __init__(self):
        self.alive = True
        self.image = CAR_IMG
        self.width = CAR_SIZE[0]
        self.height = CAR_SIZE[1]
        self.x = ROAD_WIDTH/2 - self.width/2
        self.y = ROAD_HEIGHT - self.height - 10
        self.hitbox = Rect(self.x, self.y, self.width, self.height)
        self.score = 0

    def move(self, output):
        if output > 0:
            x_movement = self.SPEED
        else:
            x_movement = - self.SPEED
        
        self.x += x_movement

        self.hitbox = self.hitbox.move(x_movement, 0)
        self.score += 1

    def data(self, obstacle):
        data = [0,0,0,0]

        data[0] = self.x                                               # Distance between car and left edge
        data[1] = ROAD_WIDTH + self.width - self.x                     # Distance between car and right edge
        data[2] = abs(self.x + self.width - obstacle.x)                # Distance right side of car with left side box                         
        data[3] = abs(self.x - obstacle.x - obstacle.width)            # Distance left side of car with right side box

        return data

    def render(self):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle():

    def __init__(self):
        self.width = OBSTACLE_SIZE[0]
        self.height = OBSTACLE_SIZE[1]
        self.x = random.randint(0,ROAD_WIDTH-self.width)
        self.y = -self.height
        self.hitbox = Rect(self.x, self.y, self.width, self.height)
        self.speed = 20

    def move(self):
        self.y += self.speed
        
        if self.y > ROAD_HEIGHT:
            self.x = random.randint(0,ROAD_WIDTH-self.width)
            self.y = -self.height
            self.hitbox = Rect(self.x, self.y, self.width, self.height)
        else:
            self.hitbox = self.hitbox.move(0, self.speed)


    def render(self):

        #to use obstacle.png:
        SCREEN.blit(OBSTACLE_IMG, (self.x, self.y))

        #to use a rectangle:
        #pygame.draw.rect(SCREEN, (80,250,0), self.hitbox)


def update_display(cars, obstacle):
    SCREEN.fill(GRAY, ROAD)

    for index, car in enumerate(cars):
        if car.alive:
            car.render()

    
    obstacle.render()
    
    # Update the display (apply the changes)
    pygame.display.update([ROAD])



def eval_genomes(genomes, config):
        
    """
    runs the simulation of the current population of
    birds and sets their fitness based on the distance they
    reach in the game.
    """
    
    global GEN
    GEN += 1
    obstacle = Obstacle()
    tick = 0
    score = 0
    FPS = 30
    fpsClock = pygame.time.Clock()


    # start by creating lists holding the genome itself, the
    # neural network associated with the genome and the
    # bird object that uses that network to play
    nets = []
    cars = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        cars.append(Car())
        ge.append(genome)


    # Game loop.
    run = True
    while run and len(cars) > 0:
        
        for event in pygame.event.get():
            if event.type == QUIT:
                run = False
                pygame.quit()
                sys.exit()

        #keys = pygame.key.get_pressed()


        obstacle.move()

        for x, car in enumerate(cars):  # give each bird a fitness of 0.1 for each frame it stays alive
            ge[x].fitness += 0.1

            output = nets[x].activate(car.data(obstacle))
            moving = output.index(max(output))
            car.move(moving)

            if obstacle.y > car.y + car.height:
                    ge[x].fitness += 5

            if not (0 < car.x < ROAD_WIDTH - car.width):
                ge[x].fitness -= 3
                car.alive = False
                cars.pop(x)
                nets.pop(x)
                ge.pop(x)

            #Check for collision with obstacles
            if car.alive and car.hitbox.colliderect(obstacle.hitbox):
                ge[x].fitness -= 3
                car.alive = False
                cars.pop(x)
                ge.pop(x)
                nets.pop(x)
            


        update_display(cars, obstacle)
        
        fpsClock.tick(FPS)



def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to a certain number of generations.
    winner = p.run(eval_genomes, 20)


if __name__ == "__main__":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)

    
    

    #for debugging purposes
    """
    with cProfile.Profile() as pr:
        run(config_path)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    """
    