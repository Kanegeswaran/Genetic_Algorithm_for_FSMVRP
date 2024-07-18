from customer import Customer
from vehicle import Vehicle
from graph import Graph
import random
import math

class GeneticAlgorithm:
    """
    Class to represent a Genetic Algorithm for solving the Vehicle Routing Problem (VRP).
    """
    def __init__(self, graph: Graph, customers: list[Customer], vehicles, depot: Customer, population_size=100, generations=1000, 
                  crossover_rate = 0.8, selection_rate=0.5, mutation_rate=0.1):
        """
        Initialize a GeneticAlgorithm object.

        :param graph: The graph representing adjacent matrix of distances between customers and depot
        :param customers: A list of customers
        :param vehicles: A list of vehicles
        :param population_size: The number of routes in the population
        :param generations: The number of generations to run the algorithm
        :param crossover_rate: The rate at which crossover occur
        :param selection_rate: The rate at which selection occur
        :param mutation_rate: The rate at which mutations occur
        """
        self.graph = graph
        self.customers = customers
        self.vehicles = vehicles
        self.depot = depot
        self.population_size = population_size
        self.generations = generations
        self.selection_rate = selection_rate
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def create_individual(self) -> tuple[list[Customer], dict[int, Vehicle]]:
        """
        Generate individual for initial population randomly.
        """
        individual = random.sample(self.customers, len(self.customers))
        return self.split_genes(individual)

    def create_population(self) -> list[tuple[list[Customer], dict[int, Vehicle]]]:
        """
        Create initial population.
        """
        # find the initial solution using Clarke Wright Saving algorithm for each vehicle type
        h_population1 = [self.savings_algo(v) for v in self.vehicles]
        # find the initial solution using Gillett Miller Sweep algorithm for each vehicle type
        h_population2 = [self.sweep_algo(v) for v in self.vehicles]

        # the initial solutions from the heuristics algorithms added with randomly generated solutions as initial population
        return h_population1 + h_population2 + [self.create_individual() for _ in range(self.population_size - len(h_population1) - len(h_population2))]
        # return [self.create_individual() for _ in range(self.population_size)]

    def calc_savings(self) -> list[tuple[float, Customer, Customer]]:
        """
        Calculate the savings of every pair of customers according to Saving Algorithm.

        :return: A list of sorted pair of customer in descending order of saved cost
        """
        
        savings = []
        for i in range(len(self.customers)):
            for j in range(i + 1, len(self.customers)):
                saving = self.graph.calc_dist(self.depot, self.customers[i]) + \
                         self.graph.calc_dist(self.depot, self.customers[j]) - \
                         self.graph.calc_dist(self.customers[i], self.customers[j])
                savings.append((saving, self.customers[i], self.customers[j]))
        return sorted(savings, key=lambda x: x[0], reverse=True)
    
    def savings_algo(self, vehicle: Vehicle) -> tuple[list[Customer], dict[int, Vehicle]]:
        """
        Perform the Saving Algorithm.
        """
        routes = []
        # creating list of routes, as each route visit a customer
        for cus in self.customers:
            routes.append([cus])
        
        # Calculate the savings of every pair of customers
        savings = self.calc_savings()

        for saving, i, j in savings:
            route_i = None
            route_j = None

            # retrieving the route of the pair from the list of routes generated initially
            for route in routes:
                if route[0] == i or route[-1] == i:
                    route_i = route
                if route[0] == j or route[-1] == j:
                    route_j = route
          
            # combining the route of pairs of customers, if the route still exist
            if route_i and route_j and route_i != route_j and sum(r.demand for r in route_i) + sum(r.demand for r in route_j) <= vehicle.capacity:
                if route_i[-1] == i and route_j[0] ==j:
                    merged_route = route_i + route_j
                    routes[routes.index(route_i)] = merged_route
                    routes.remove(route_j)

        # assign the solution into individual 
        genes = ([], {})
        i = 0
        for route in routes:
            genes[0].extend(route)
            genes[1].update({len(route)+i:vehicle})
            i += len(route)
        
        return genes
    
    def calculate_polar_angle(self, customer: Customer) -> float:
        """
        Calculate the angle of from depot according to Sweep Algorithm.

        :return: the angle of a customer from depot
        """
        y_diff = customer.latitude - self.depot.latitude
        x_diff = customer.longitude - self.depot.longitude
        angle = math.atan2(y_diff, x_diff) * 180 / math.pi
        return angle if angle >= 0 else angle + 360
    
    def sweep_algo(self, vehicle: Vehicle) -> tuple[list[Customer], dict[int, Vehicle]]:
        """
        Perform the Sweep Algorithm.
        """
        # sort the customers by the customer having the smallest polar angle from depot
        customers = sorted(self.customers, key=lambda customer: self.calculate_polar_angle(customer))
        
        genes = ([], {})
        current_route = []
        current_demand = 0
        
        for i, customer in enumerate(customers):
            # appending the customer from the individual to the vehicle route and make sure that the capacity of selected vehicle doesnt exceeded
            if current_demand + customer.demand <= vehicle.capacity:
                current_route.append(customer)
                current_demand += customer.demand
            else:
                genes[0].extend(current_route)
                genes[1].update({i:vehicle})
                # if the capacity of the vehicle is full, assign next customer to new route of new randomly selected vehicle
                current_route = [customer]
                current_demand = customer.demand

        #if the last route exist as it may not full the capcity of the vehicle, insert it into the individual
        if current_route:
            genes[0].extend(current_route)
            genes[1].update({i+1:vehicle})

        return genes

    def split_genes(self, individual: list[Customer]) -> tuple[list[Customer], dict[int, Vehicle]]:
        """
        Assign vechicles for randomly generated individuals.
        """
        genes = ([], {})
        current_route = []
        current_demand = 0
        # randomly choosing a vehicle type
        current_vehicle = random.choice(self.vehicles)

        for i,customer in enumerate(individual):
            # appending the customer from the individual to the vehicle route and make sure that the capacity of selected vehicle doesnt exceeded
            if current_demand + customer.demand <= current_vehicle.capacity:
                current_route.append(customer)
                current_demand += customer.demand
            else:
                if current_route:
                    genes[0].extend(current_route)
                    genes[1].update({i:current_vehicle})
                # if the capacity of the vehicle is full, assign next customer to new route of new randomly selected vehicle
                current_route = [customer]
                current_demand = customer.demand
                current_vehicle = random.choice(self.vehicles)
        
        #if the last route exist as it may not full the capcity of the vehicle, insert it into the individual
        if current_route:
            genes[0].extend(current_route)
            genes[1].update({i+1:current_vehicle})

        return genes

    def fitness(self, individual:tuple[list[Customer], dict[int, Vehicle]]) -> float:
        """
        Evaluate the fitness of each route in the population.
        """
        genes, routes = individual
        total_cost = 0
        i = 0

        # calculating the total cost the routes of all vehicles in the individual
        for route in routes:
            total_cost += self.graph.calc_route_cost(genes[i:route], routes[route])
            i = route
        return 1 / total_cost # Higher fitness for lower cost
        # return round(((1 / total_cost) * 100), 2)

    def tournament_selection(self, population: list[tuple[list[Customer], dict[int, Vehicle]]], tournament_size=3) -> tuple[list[Customer], dict[int, Vehicle]]:
        """
        Select parents for crossover using a tournament selection method.

        :return: selected parent routes with highest fitness score
        """

        # randomly choose three individual from the population and return the best one
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=self.fitness)
    
    def crossover(self, parent1: tuple[list[Customer], dict[int, Vehicle]], parent2: tuple[list[Customer], dict[int, Vehicle]]) -> tuple[list[Customer], dict[int, Vehicle]]:
        """
        Perform crossover between two parents to create a child route.

        :param parent1: The first parent route
        :param parent2: The second parent route
        :return: A child route resulting from the crossover
        """

        # make sure that the crossover rate achieved
        if random.random() > self.crossover_rate:
            return parent1

        # select a random crossover point from its routes
        crossover_point = random.randint(1, len(parent1[0]) - 1)
        # slice the parent1 till the crossover point adn assign to child
        child = parent1[0][:crossover_point]
        # add the missed out customers from parent2 into child
        for customer in parent2[0]:
            if customer not in child:
                child.append(customer)
        return self.split_genes(child)

    def mutate(self, individual: tuple[list[Customer], dict[int, Vehicle]]) -> tuple[list[Customer], dict[int, Vehicle]]:
        """
        Mutate a individual with a given mutation rate by shuffling route of a randomly selected vechicle in the individual.

        :param route: The mutated individual
        """

        # make sure that the mutation rate achieved
        if random.random() < self.mutation_rate:
            genes, routes = individual
            mutated_genes = []
            i = 0
            # randomly generate a int between 0 to total number of vehicles in the individual to choose one vehicle randomly
            r = random.randint(0,len(routes))
            for j, route in enumerate(routes):
                new_genes = genes[i:route]
                i = route
                if r==j:
                    # shuffle the route of the chosen vehicle
                    random.shuffle(new_genes)
                mutated_genes.extend(new_genes)
            return (mutated_genes, routes)
        return individual

    def run(self) -> tuple[list[Customer], dict[int, Vehicle]]:
        """
        Run the genetic algorithm for a specified number of generations.

        :return: The best solution found by the algorithm
        """
        # create initial population
        population = self.create_population()
        # print("population 0 : \n", population, "\n")

        # run the genetic algorithm for (default = 1000) generations
        for i in range(self.generations):
            # sort the population by the population having the highest fitness score
            population = sorted(population, key=self.fitness, reverse=True)
            # select selection_rate*population size of individuals that have the highest fitness score
            population = new_population = population[:int(self.selection_rate*self.population_size)]  # Elitism

            # Do crossover and mutation in the selected population to replace the neglected individuals
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            population = new_population
            # local_best = max(population, key=self.fitness)
            # print("best score (", i, "): ", self.fitness(local_best))
            # avg = sum(self.fitness(individual) for individual in population)/self.population_size
            # print("Avg score (", i, "): ", round(avg, 2))

            # print("population ", i , ": \n", population, "\n")

        best_individual = max(population, key=self.fitness)
        return best_individual
