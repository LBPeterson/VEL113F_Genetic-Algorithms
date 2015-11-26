# VEL113F_Genetic-Algorithms
Final project for Design Optimization Course
Attempt to find the shortest route between all of these locations in order to estimate local fish populations:
![Locations](/Presentation/locations.png)
Several factors should be taken into account, including needing to sail around land for some points as well as Icelandic water currents.


After running a genetic algorithm utilizing an edge recombination scheme, reciprocal mutation, and using the Lin-Kernighan heuristic for 20000 generations with 20 individuals, a solution is presented:
![Final](/Presentation/final.png)
More generations will be needed to find the optimal solution, and a rewrite of the edge recombination scheme with an emphasis on efficiency would improve the algorithm quite a lot. 

The ER-LIN* files are the output of the program to create this solution.
