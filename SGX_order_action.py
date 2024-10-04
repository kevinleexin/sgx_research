from enum import IntEnum


class Action(IntEnum):
    AddOrder = 65,
    FirstOrderExecuted = 67,
    OrderDelete = 68,
    SecondOrderExecuted = 69,
    TickSize = 76,
    CombinationOrderBookDirectory = 77,
    OrderBookState = 79,
    TradeMessageIdentifier = 80,
    OrderBookDirectory = 82,
    SystemEvent = 83,
    Seconds = 84,
    OrderReplace = 85,
    EquilibriumPriceUpdate = 90,


if __name__ == "__main__":
    print(Action.AddOrder)
