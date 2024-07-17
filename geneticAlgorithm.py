from customer import Customer
from vehicle import Vehicle
from graph import Graph
import random
import math

class GeneticAlgorithm:
    def __init__(self, graph: Graph, customers: list[Customer], vehicles, depot: Customer, population_size=100, generations=1000, 
                  crossover_rate = 0.8, selection_rate=0.5, mutation_rate=0.1):
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
        individual = random.sample(self.customers, len(self.customers))
        return self.split_genes(individual)

    def create_population(self) -> list[tuple[list[Customer], dict[int, Vehicle]]]:
        h_population1 = [self.savings_algo(v) for v in self.vehicles]
        h_population2 = [self.sweep_algo(v) for v in self.vehicles]

        return h_population1 + h_population2 + [self.create_individual() for _ in range(self.population_size - len(h_population1) - len(h_population2))]
        # return [self.create_individual() for _ in range(self.population_size)]

    def calc_savings(self) -> list[tuple[float, Customer, Customer]]:
        savings = []
        for i in range(len(self.customers)):
            for j in range(i + 1, len(self.customers)):
                saving = self.graph.calc_dist(self.depot, self.customers[i]) + \
                         self.graph.calc_dist(self.depot, self.customers[j]) - \
                         self.graph.calc_dist(self.customers[i], self.customers[j])
                savings.append((saving, self.customers[i], self.customers[j]))
        return sorted(savings, key=lambda x: x[0], reverse=True)
    
    def savings_algo(self, vehicle: Vehicle) -> tuple[list[Customer], dict[int, Vehicle]]:
        routes = [] 
        for cus in self.customers:
            routes.append([cus])
        
        savings = self.calc_savings()

        for saving, i, j in savings:
            route_i = None
            route_j = None

            for route in routes:
                if route[0] == i or route[-1] == i:
                    route_i = route
                if route[0] == j or route[-1] == j:
                    route_j = route
          
            if route_i and route_j and route_i != route_j and sum(r.demand for r in route_i) + sum(r.demand for r in route_j) <= vehicle.capacity:
                if route_i[-1] == i and route_j[0] ==j:
                    merged_route = route_i + route_j
                    routes[routes.index(route_i)] = merged_route
                    routes.remove(route_j)

        genes = ([], {})
        i = 0
        for route in routes:
            genes[0].extend(route)
            genes[1].update({len(route)+i:vehicle})
            i += len(route)
        
        return genes
    
    def calculate_polar_angle(self, customer: Customer) -> float:
        y_diff = customer.latitude - self.depot.latitude
        x_diff = customer.longitude - self.depot.longitude
        angle = math.atan2(y_diff, x_diff) * 180 / math.pi
        return angle if angle >= 0 else angle + 360
    
    def sweep_algo(self, vehicle: Vehicle) -> tuple[list[Customer], dict[int, Vehicle]]:
        customers = sorted(self.customers, key=lambda customer: self.calculate_polar_angle(customer))
        
        genes = ([], {})
        current_route = []
        current_demand = 0
        
        for i, customer in enumerate(customers):
            if current_demand + customer.demand <= vehicle.capacity:
                current_route.append(customer)
                current_demand += customer.demand
            else:
                genes[0].extend(current_route)
                genes[1].update({i:vehicle})
                current_route = [customer]
                current_demand = customer.demand

        if current_route:
            genes[0].extend(current_route)
            genes[1].update({i+1:vehicle})

        return genes

    def split_genes(self, individual: list[Customer]) -> tuple[list[Customer], dict[int, Vehicle]]:
        genes = ([], {})
        current_route = []
        current_demand = 0
        current_vehicle = random.choice(self.vehicles)

        for i,customer in enumerate(individual):
            if current_demand + customer.demand <= current_vehicle.capacity:
                current_route.append(customer)
                current_demand += customer.demand
            else:
                if current_route:
                    genes[0].extend(current_route)
                    genes[1].update({i:current_vehicle})
                current_route = [customer]
                current_demand = customer.demand
                current_vehicle = random.choice(self.vehicles)

        if current_route:
            genes[0].extend(current_route)
            genes[1].update({i+1:current_vehicle})

        return genes

    def fitness(self, individual:tuple[list[Customer], dict[int, Vehicle]]) -> float:
        genes, routes = individual
        total_cost = 0
        i = 0
        for route in routes:
            total_cost += self.graph.calc_route_cost(genes[i:route], routes[route])
            i = route
        return 1 / total_cost  # Higher fitness for lower cost

    def tournament_selection(self, population: list[tuple[list[Customer], dict[int, Vehicle]]], tournament_size=3) -> tuple[list[Customer], dict[int, Vehicle]]:
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=self.fitness)
    
    def crossover(self, parent1: tuple[list[Customer], dict[int, Vehicle]], parent2: tuple[list[Customer], dict[int, Vehicle]]) -> tuple[list[Customer], dict[int, Vehicle]]:
        if random.random() > self.crossover_rate:
            return parent1

        crossover_point = random.randint(1, len(parent1[0]) - 1)
        child = parent1[0][:crossover_point]
        for customer in parent2[0]:
            if customer not in child:
                child.append(customer)
        return self.split_genes(child)

    def mutate(self, individual: tuple[list[Customer], dict[int, Vehicle]]) -> tuple[list[Customer], dict[int, Vehicle]]:
        if random.random() < self.mutation_rate:
            genes, routes = individual
            mutated_genes = []
            i = 0
            r = random.randint(0,len(routes))
            for j, route in enumerate(routes):
                new_genes = genes[i:route]
                i = route
                if r==j:
                    random.shuffle(new_genes)
                mutated_genes.extend(new_genes)
            return (mutated_genes, routes)
        return individual

    def run(self) -> tuple[list[Customer], dict[int, Vehicle]]:
        population = self.create_population()
        # print("population 0 : \n", population, "\n")

        for _ in range(self.generations):
            population = sorted(population, key=self.fitness, reverse=True)
            population = new_population = population[:int(self.selection_rate*self.population_size)]  # Elitism

            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population
            # print("population ", _ , ": \n", population, "\n")

        best_individual = max(population, key=self.fitness)
        return best_individual
