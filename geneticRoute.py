from customer import Customer
from vehicle import Vehicle
from graph import Graph
from geneticAlgorithm import GeneticAlgorithm
import random
import matplotlib.pyplot as plt

def read_customers(file_path: str) -> list[Customer]: 
    """
    Read customer data from a file and return a list of Customer objects.

    :param file_path: Path to the file containing customer data
    :return: List of Customer objects
    """
    customers = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip():
                customer_data = line.split(', ')
                customer = Customer(
                    int(customer_data[0]), 
                    float(customer_data[1]), 
                    float(customer_data[2]), 
                    int(customer_data[3])
                )
                customers.append(customer)
    return customers

def read_vehicles(file_path: str) -> list[Vehicle]:
    """
    Read vehicle data from a file and return a list of Vehicle objects.

    :param file_path: Path to the file containing vehicle data
    :return: List of Vehicle objects
    """
    vehicles = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip():
                vehicle_data = line.split(', ')
                vehicle = Vehicle(
                    vehicle_data[0],
                    int(vehicle_data[1]),
                    float(vehicle_data[2])
                )
                vehicles.append(vehicle)
    return vehicles

if __name__ == "__main__" :
    """
    Main block to execute the genetic algorithm for vehicle routing.
    """
    random.seed(9)

    # Reading customers and vehicles from input files
    customers = read_customers("C:\\Users\\kanag\\Desktop\\modefair\\customers_input.txt")
    vehicles = read_vehicles("C:\\Users\\kanag\\Desktop\\modefair\\vehicles_input.txt")

    # Creating a graph with customers
    graph = Graph(customers)

    # Extracting the depot (first customer) from the list
    depot = customers.pop(0)

    # Initializing and running the genetic algorithm
    ga = GeneticAlgorithm(graph, customers, vehicles, depot)
    best_solution = ga.run()
    genes, routes = best_solution
    total_dist = 0
    total_cost = 0
    trips = []
    i = 0

    # Calculating total distance and cost for the best solution
    for route in routes:
        current_customers, current_vehicle = genes[i:route], routes[route]
        i = route
        total_cost += graph.calc_route_cost(current_customers, current_vehicle)
        trip = graph.route_dist(current_customers)
        total_dist += sum(trip[0])
        trips.append((current_vehicle, trip, current_customers))

    # Plotting the results
    plt.figure(figsize=(10,5))
    plt.plot(ga.best_fitness_per_generation, label='Best Fitness')
    plt.plot(ga.avg_fitness_per_generation, label='Average Fitness')
    plt.xlabel('Generation')
    plt.xlabel('Fitness')
    plt.title('Best Fit and Average Fit vs Generation')
    plt.legend()
    plt.show()

    # Printing the results
    print(f"Total Distance = {total_dist:.3f} km")
    print(f"Total Cost = RM {total_cost:.2f}\n")

    for i, (current_vehicle, trip, current_customers) in enumerate(trips):
        print(f"Vehicle {i+1} ({current_vehicle.vehicleType}):")
        current_dist = sum(trip[0])
        print(f"Round Trip Distance: {current_dist:.3f} km, Cost: RM {(current_dist*current_vehicle.cost_per_km)}, Demand: {trip[1]}")
        print("Depot -> ", end="")
        for c in range(len(current_customers)):
            print(f"C{current_customers[c].customerId} ({trip[0][c]:.3f} km) -> ", end='')
        print(f"Depot ({trip[0][c+1]:.3f}) km\n")
        
