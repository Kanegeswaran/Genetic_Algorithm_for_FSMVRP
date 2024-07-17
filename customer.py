class Customer:

    """
    Class to represent a Customer with an ID, location (latitude and longitude), and demand.
    """

    def __init__(self, customerId: int, latitude: float, longitude: float, demand: int) -> None:
        """
        Initialize a Customer object.

        :param customerId: ID of the customer
        :param latitude: Latitude coordinate of the customer's location
        :param longitude: Longitude coordinate of the customer's location
        :param demand: Demand value of the customer
        """
        self.customerId = customerId
        self.latitude = latitude
        self.longitude = longitude
        self.demand = demand

    def __str__(self) -> str:
        """
        String representation of the Customer object.

        :return: A string that represents the customer
        """
        return f"Customer {self.customerId}"
    
    def __repr__(self) -> str:
        """
        Representation of the Customer object for debugging.

        :return: A string that represents the customer
        """
        return self.__str__()
