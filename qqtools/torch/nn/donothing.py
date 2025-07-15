class DoNothing:
    def passby(self, *args, **kwargs):
        pass

    def __getattr__(self, *args):
        return self.passby
