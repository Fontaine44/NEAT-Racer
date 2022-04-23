# NEAT Racer

## About

An AI coded in Python that plays an "endless runner" style game. It uses the NEAT-Python module for the simulation and the Pygame module for the front-end. This is my first complete project in Python and I really enjoyed the whole process of development.

**This project is a work in progress...**

### Built With

* [Pygame](https://www.pygame.org/wiki/about)
* [NEAT-Python](https://neat-python.readthedocs.io/en/latest/neat_overview.html)

## Simulation
![Simulation gif](https://media.giphy.com/media/qiK98Je5982HjAGlWH/giphy.gif)

## NEAT algorithm

NEAT (NeuroEvolution of Augmenting Topologies) is "an evolutionary algorithm that creates artificial neural networks".  It starts with very simple networks and complexifies them over time by adding new neurons and connections. Kenneth Stanley is the main researcher behind this algorithm and you can find more information about it [here](http://www.cs.ucf.edu/~kstanley/neat.html).

Inputs:

 1. Distance between the car and the left edge
 2. Distance between the car and the right edge
 3. Distance between the right side of the car and the left side of the obstacle
 4. Distance between the left side of the car and the right side of the obstacle

Output:
I chose the sigmoid activation function.\
![Sigmoid Graph](https://neat-python.readthedocs.io/en/latest/_images/activation-sigmoid.png)


## Getting Started

Simply run NEAT_Racer.py with the following prerequisites installed on your system.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* Python 3 (I used version 3.9.5 but I guess that it would work with other versions)
* Pygame
  ```
  pip install pygame
  ```
* NEAT-Python
  ```
  pip install neat-python
  ```

**This project is a work in progress...**
