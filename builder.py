class Builder:
    def __init__(self, parent_builder=None):
        self.parent_builder = parent_builder

    def build(self, callback=True, counter=0):
        if callback and self.parent_builder is not None:
            self.parent_builder.register(self)
            return self.parent_builder
        else:
            return self.build_impl(counter)

    def build_impl(self, counter):
        raise NotImplementedError

    def register(self, child_builder):
        raise NotImplementedError


"""
SimulationBuilder(configpath, configname)
    .beacon_node(4)
        .debug(True)
        .profile(True)
        .validators(7)
            .keydir(keydir)
            .build()
        .build()
    .build()
"""
