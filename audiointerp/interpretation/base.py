from abc import ABC, abstractmethod


class BaseInterpretation(ABC):

    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device

    @abstractmethod
    def interpret(self, inputs, **kwargs):
        """Apply interpretation method to given samples"""
        pass


