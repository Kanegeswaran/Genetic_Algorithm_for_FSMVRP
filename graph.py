import math
import numpy as np
from customer import Customer
from vehicle import Vehicle

class Graph:
    """
    Class to represent a graph with customers, depot and distances between them.
    """
    def __init__(self, customers: list[Customer]) -> None :
        """
        Initialize a Graph object.

        :param customers: A list of customers
        """
        self.depot = customers[0]
        self.customers = customers
        self.createAdjMatrix(len(customers))

    def calc_dist(self, cust1: Customer, cust2: Customer) -> float:
        """
        Calculate distance between two customers

        :param cust1: Source (customer 1)
        :param cust2: Destination (customer two)
        """
        return 100 * math.sqrt((cust2.longitude - cust1.longitude)**2+(cust2.latitude - cust1.latitude)**2)

    def createAdjMatrix(self, size: int) -> None:
        """
        Create the adjacent matrix of distance between customers

        :param size: Total number of customers
        """
        self.adjMatrix = np.zeros((size,size))
        for i in range(size):
            for j in range(i+1, size):
                self.adjMatrix[i][j] = self.adjMatrix[j][i] = self.calc_dist(self.customers[i], self.customers[j])
        
    def route_dist(self, route: list[Customer]) -> list :
        """
        Calculate the distances between a list of customers

        :param route: A list customers in the route
        """
        dist = [self.adjMatrix[0][route[0].customerId]]
        total_demand = route[0].demand
        
        for i in range(len(route)-1):
            dist.append(self.adjMatrix[route[i].customerId][route[i+1].customerId])
            total_demand += route[i+1].demand

        dist.append(self.adjMatrix[route[-1].customerId][0])
     
        return (dist, total_demand)
    
    def calc_route_cost(self, route: list[Customer], vehicle: Vehicle) -> float :
        """
        Calculate the total cost of the route travelled by a vehicle 

        :param route: A list customers in the route
        :param vehicle: The vehicle that travelled the route
        """
        total_dist = self.adjMatrix[route[-1].customerId][0] + self.adjMatrix[0][route[0].customerId]
        total_demand = route[0].demand

        for i in range(len(route)-1):
            total_dist += self.adjMatrix[route[i].customerId][route[i+1].customerId]
            total_demand += route[i+1].demand

        if(total_demand > vehicle.capacity):
            return float('inf')
        
        return total_dist * vehicle.cost_per_km
    
    def total_demand(route: list[Customer]) -> int:
        """
        Calculate the total demand of customers in a route

        :param route: A list customers in the route
        """
        return sum(customer.demand for customer in route)
    
    def __str__(self) -> str:
        """
        String representation of the graph object.

        :return: A string that represents the graph
        """
        return f'{self.adjMatrix}'




