class Customer:

    def __init__(self, customerId: int, latitude: float, longitude: float, demand: int) -> None:
        self.customerId = customerId
        self.latitude = latitude
        self.longitude = longitude
        self.demand = demand

    def __str__(self) -> str:
        return f"Customer {self.customerId}"
    
    def __repr__(self) -> str:
        return self.__str__()