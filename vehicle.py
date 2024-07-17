class Vehicle:

    def __init__ (self, vehicleType: str, capacity: int, cost_per_km: float) -> None :
        self.vehicleType = vehicleType
        self.capacity = capacity
        self.cost_per_km = cost_per_km

    def __str__(self) -> str:
        return f"Vehicle {self.vehicleType}"
    
    def __repr__(self) -> str:
        return self.__str__()
