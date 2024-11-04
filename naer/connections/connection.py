from inspect import getsource
from textwrap import dedent
import ast
from naer.renderer.cstyle import CStyleRenderer


def naer_random():
    return 0.5


def Connection(connection_type):
    class modified(connection_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def reflect(self):
            print("Transpiling result:")
            source = dedent(getsource(self.connect))
            ast_module = ast.parse(source)
            renderer = CStyleRenderer(ast_module)
            renderer.visit(ast_module)
            print(renderer.output)

    return modified
