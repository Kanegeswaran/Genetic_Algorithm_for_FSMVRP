# Genetic Algorithm for Fleet Size and Mix Vehicle Routing problem (FSMVRP)
 
## Overview
This project aims to solve the variant of Vehicle Routing Problem, Fleet Size and Mix Vehicle Routing problem using a metaheuristic algorithm which is Genetic Algorithm. The objective is to optimize the routing of a fleet of vehicles to efficiently deliver goods to various customer locations at the lowest cost while ensuring that all delivery locations are visited and all demands are met.

## Problem Description
### Objective: 
- Optimize delivery routes to minimize total cost.
### Constraints:
- Each delivery location must be visited exactly once.
- The total demand of each vehicle route must not exceed its maximum capacity.
### Assumptions:
- Vehicles start and end their routes at the same depot location.
- Each vehicle only travels one round trip.
- There is no limit on the number of vehicles.
- Travel times between any two locations are the same in both directions.
- Deliveries can be made at any time, with no time windows for deliveries.
- Vehicle travel distance is calculated using the Euclidean distance formula.

## Metaheuristic Algorithm
A Genetic Algorithm is implemented to solve the FSMVRP. The Genetic Algorithm uses operations such as selection, crossover, and mutation to evolve a population of solutions over several generations, aiming to find the optimal routing solution.

