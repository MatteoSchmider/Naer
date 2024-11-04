from naer.renderer.renderer import Renderer
import ast


class CStyleRenderer(Renderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth = 0

    def generic_visit(self, node):
        test = "encountered unsupported node type: " + type(node).__name__ + "\n"
        test += "ast parsed as: \n"
        test += ast.dump(node, indent=4)
        raise NotImplementedError(test)

    def visit_Attribute(self, node: ast.Attribute):
        return self.visit(node.value) + "." + node.attr

    def visit_Name(self, node: ast.Name):
        return str(node.id)

    def visit_Call(self, node: ast.Call):
        out = self.visit(node.func) + "("
        for arg in node.args:
            out = self.visit(arg) + ","
        return out[:-1] + ")"

    def visit_Constant(self, node: ast.Constant):
        return str(node.value)

    def visit_Module(self, node: ast.Module):
        return self.visit(node.body[0])

    def visit_FunctionDef(self, node: ast.FunctionDef):
        out = ""
        for statement in node.body:
            out += self.visit(statement)
        return out

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) != 1:
            raise NotImplementedError(
                "Assign with more than one target is not supported in C-Style"
            )
        return self.visit(node.targets[0]) + " = " + self.visit(node.value) + ";\n"

    def visit_AnnAssign(self, node: ast.AnnAssign):
        return self.visit(node.target) + " = " + self.visit(node.value) + ";\n"

    def visit_AugAssign(self, node: ast.AugAssign):
        return self.visit(node.target) + " " + self.visit(node.op) + "= " + self.visit(node.value) + ";\n"

    def visit_If(self, node: ast.If):
        out = "if (" + self.visit(node.test) + ") {\n"
        self.depth += 1
        for statement in node.body:
            out += "    " * self.depth + self.visit(statement)
        if node.orelse:
            out += "} else {\n"
            for statement in node.orelse:
                out += "    " * self.depth + self.visit(statement)
        self.depth -= 1
        return out + "    " * self.depth + "}\n"

    def visit_Return(self, node: ast.Return):
        return "return " + self.visit(node.value) + ";\n"

    def visit_UnaryOp(self, node: ast.UnaryOp):
        return "(" + self.visit(node.op) + self.visit(node.operand) + ")"

    def visit_USub(self, node: ast.USub):
        return "-"

    def visit_UAdd(self, node: ast.UAdd):
        return "+"

    def visit_BinOp(self, node: ast.BinOp):
        return "(" + self.visit(node.left) + " " + self.visit(node.op) + " " + self.visit(node.right) + ")"

    def visit_Add(self, node: ast.Add):
        return "+"

    def visit_Sub(self, node: ast.Sub):
        return "-"

    def visit_Mult(self, node: ast.Mult):
        return "*"

    def visit_Div(self, node: ast.Div):
        return "/"

    def visit_Mod(self, node: ast.Mod):
        return "%"

    def visit_LShift(self, node: ast.LShift):
        return "<<"

    def visit_RShift(self, node: ast.RShift):
        return ">>"

    def visit_BitOr(self, node: ast.BitOr):
        return "|"

    def visit_BitXor(self, node: ast.BitXor):
        return "^"

    def visit_BitAnd(self, node: ast.BitAnd):
        return "&"

    def visit_BoolOp(self, node: ast.BoolOp):
        if len(node.values) < 1:
            raise NotImplementedError(
                "BoolOp with no values is not supported in C-Style"
            )
        out = "(" + self.visit(node.values[0])
        for value in node.values[1:]:
            out += " " + self.visit(node.op) + " " + self.visit(value)
        return out + ")"

    def visit_And(self, node: ast.And):
        return "&&"

    def visit_Or(self, node: ast.Or):
        return "||"

    def visit_Compare(self, node: ast.Compare):
        return self.visit(node.left) + " " + self.visit(node.ops[0]) + " " + self.visit(node.comparators[0])

    def visit_Eq(self, node: ast.Eq):
        return "=="

    def visit_NotEq(self, node: ast.NotEq):
        return "!="

    def visit_Lt(self, node: ast.Lt):
        return "<"

    def visit_LtE(self, node: ast.LtE):
        return "<="

    def visit_Gt(self, node: ast.Gt):
        return ">"

    def visit_GtE(self, node: ast.GtE):
        return ">="
