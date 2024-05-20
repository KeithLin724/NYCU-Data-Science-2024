from base import Track, Config


class Trainer(Track):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.build_model()

    def build_model(self):
        return

    def train(self):
        return
