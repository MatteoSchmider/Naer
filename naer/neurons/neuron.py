from inspect import getsource
from math import exp
from textwrap import dedent
from copy import deepcopy
import ast
from naer.renderer.cstyle import CStyleRenderer


class NeuronAnalyzer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        # we need three outputs here:
        # 1. list of names + naer types + values of all used variables
        #   => have two maps: name -> value and name -> naer_type
        #   => ast walk and replace all names + attribute_names by their canonical names
        # 2. reduced update AST with all variable names normalized as names
        # 3. reduced receive AST with all variable names normalized as names
        # a renderer does not need to know about 1.) to render 3.) if we ensure correctness in here
        # => makes renderer more light weight and generic!


def naer_exp(x):
    return exp(x)


def naer_random():
    return 5


def Neuron(neuron_type):
    class modified(neuron_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def reflect(self):
            print("Transpiling result:")
            source = dedent(getsource(self.receive))
            ast_module = ast.parse(source)
            renderer = CStyleRenderer(self)
            out = renderer.visit(ast_module)
            print(out)

        def print_values(self):
            print(vars(self))

        def print_types(self):
            pass

    return modified
