class Vehicle:
    """
    Class to represent a Vehicle with a type, capacity, and cost per kilometer.
    """

    def __init__(self, vehicleType: str, capacity: int, cost_per_km: float) -> None:
        """
        Initialize a Vehicle object.

        :param vehicleType: Type of the vehicle
        :param capacity: Capacity of the vehicle
        :param cost_per_km: Cost per kilometer for the vehicle
        """
        self.vehicleType = vehicleType
        self.capacity = capacity
        self.cost_per_km = cost_per_km

    def __str__(self) -> str:
        """
        String representation of the Vehicle object.

        :return: A string that represents the vehicle
        """
        return f"Vehicle {self.vehicleType}"
    
    def __repr__(self) -> str:
        """
        Representation of the Vehicle object for debugging.

        :return: A string that represents the vehicle
        """
        return self.__str__()

