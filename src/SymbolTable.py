from SymbolEntry import *
from Node import VarNode, FunctionNode
class SymbolTable:
    def __init__(self) -> None:
        self.table: list[SymbolEntry] = []

    def lookup(self, in_object: object | str):
        """
        Search for the object in the table
        :param in_object: the object
        :return: SymbolEntry or raise error
        """
        matching = []
        if isinstance(in_object, object):
            for entry in self.table:
                if entry.object == in_object:
                    matching.append(entry)
        if isinstance(in_object, str):
            for entry in self.table:
                if entry.name == in_object:
                    matching.append(entry)
        return matching

    def exists(self, in_object: object | str) -> bool:
        if len(self.lookup(in_object)) == 0:
            return False
        return True
    def insert(self, in_object: SymbolEntry, index: int = 0) -> None:
        """
        Insert symbol table entry with default index 0
        :param index: indicates where to insert the object
        :type in_object: SymbolEntry
        """
        self.table.insert(index, in_object)

    def update(self, in_object: VarNode | FunctionNode) -> bool:
        if not (isinstance(in_object, VarNode) or isinstance(in_object, FunctionNode)):
            return False
        for entry in self.table:
            if entry.object == in_object:
                entry.object = in_object
                entry.type = in_object.type
                entry.const = in_object.const
                return True

    def remove(self, in_object: SymbolEntry) -> None:
        """
        Remove an object from the table
        :param in_object: object that needs to be removed
        """
        self.table.remove(in_object)

