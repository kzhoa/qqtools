class DoNothing:
    def __init__(self, *args, **kwargs):
        pass

    def passby(self, *args, **kwargs):
        pass

    def __getattr__(self, *args):
        return self.passby


class Donothing(DoNothing):
    """Poka-yoke"""

    pass
