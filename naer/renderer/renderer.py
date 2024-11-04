import ast


class Renderer(ast.NodeTransformer):
    def __init__(self, type_to_render):
        super().__init__()
        self.type_to_render = type_to_render

    def render_update(self):
        pass

    def render_receive(self):
        pass
