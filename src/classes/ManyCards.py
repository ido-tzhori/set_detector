import numpy as np
from . import Card
from itertools import combinations

class ManyCards:
    def __init__(self, cards):
        self.cards = cards
        self.sets = []

    def __str__(self):
        return f"num of sets: {len(self.sets)}"
    
    def is_set(self, card1, card2, card3):
        # Check if any card count is 0
        if card1.count == 0 or card2.count == 0 or card3.count == 0:
            return False
        # Check if the color is a set
        if (card1.color == card2.color and card2.color == card3.color) or \
        (card1.color != card2.color and card2.color != card3.color and card1.color != card3.color):
            # Check if the number of shapes is a set
            if (card1.count == card2.count and card2.count == card3.count) or \
            (card1.count != card2.count and card2.count != card3.count and card1.count != card3.count):
                # Check if the shade is a set
                if (card1.shade == card2.shade and card2.shade == card3.shade) or \
                (card1.shade != card2.shade and card2.shade != card3.shade and card1.shade != card3.shade):
                    # Check if the shape is a set
                    if (card1.shape == card2.shape and card2.shape == card3.shape) or \
                    (card1.shape != card2.shape and card2.shape != card3.shape and card1.shape != card3.shape):
                        return True
        return False


    def return_all_sets(self):
        combos = list(combinations(self.cards, 3))
        sets = []
        for combo in combos:
            if self.is_set(*combo):
                sets.append(list(combo))
        self.sets = sets
        return self

    def multiple(self):
        r = []
        seen = {}
        for s in self.sets:
            tup_set = []
            for card in s:
                if card in seen:
                    tup_set.append((card, seen[card] + 1))
                    seen[card] += 1
                else:
                    tup_set.append((card, 0))
                    seen[card] = 0
            r.append(tup_set)
        self.sets = r

        return self