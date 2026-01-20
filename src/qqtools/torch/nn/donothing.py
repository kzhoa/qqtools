class DoNothing:
    def __init__(self, *args, **kwargs):
        pass

    def passby(self, *args, **kwargs):
        pass

    def __getattr__(self, *args):
        return self.passby

    def __enter__(self, *args, **kwargs):
        pass

    def __exit__(self, *args, **kwargs):
        pass


Donothing = DoNothing  # Poka-yoke


def donothing(*args, **kwargs):
    return
