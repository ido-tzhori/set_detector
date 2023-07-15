class Card:
    def __init__(self, number, symbol, shading, color):
        self.number = number
        self.symbol = symbol
        self.shading = shading
        self.color = color

    def __str__(self):
        return f"Card({self.number}, {self.symbol}, {self.shading}, {self.color})"

card = Card("two", "squiggle", "solid", "green")

print(card)