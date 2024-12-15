from .node import VarNode


class Manager:  # Manager class for register management using LRU
    """
    Manager class for register management using LRU
    Head is the least recently used register
    Tail is the most recently used register
    """

    def __init__(self, size) -> None:
        self.head = None
        self.tail = None
        self.size = size

    def add(self, register):
        """
        Adds a register to the manager
        :param register: the register to be added
        :return:
        """
        if self.head is None:
            self.head = register
            self.tail = register
        else:
            self.tail.next = register
            register.prev = self.tail
            self.tail = register

    def remove(self):
        """
        Removes the least recently used register
        :return: register
        """
        if self.head is None:
            return None
        register = self.head
        self.head = self.head.next
        self.head.prev = None
        return register

    def LRU(self, in_object):
        return

    def LRU_delete(self, register_name: str):
        return

    def search(self, in_object):
        temp_head = self.head
        while temp_head is not None:
            if temp_head.object is None:
                temp_head = temp_head.next
                continue
            if isinstance(in_object, VarNode):
                if temp_head.object == in_object:
                    in_object.register = temp_head
                    return temp_head
            else:
                if temp_head.object.key == in_object.value:
                    in_object.register = temp_head
                    return temp_head
            temp_head = temp_head.next
        return None

    def clear(self):
        return

    def shuffle(self, in_reg=None):
        if in_reg == self.tail:
            return
        if in_reg is None or in_reg == self.head:
            newHead = self.head.next
            self.tail.next = self.head
            self.head.prev = self.tail
            self.tail = self.head
            self.tail.next = None
            newHead.prev = None
            self.head = newHead
        else:
            # replace a node in the middle of the chain by moving it to the tail
            if in_reg.prev is not None:
                in_reg.prev.next = in_reg.next
            if in_reg.next is not None:
                in_reg.next.prev = in_reg.prev
            self.tail.next = in_reg
            in_reg.prev = self.tail
            self.tail = in_reg
            in_reg.next = None

    def shuffle_name(self, in_reg_name: str):
        tempHead = self.head
        while tempHead is not None:
            if tempHead.name == in_reg_name:
                self.shuffle(tempHead)
                break
            tempHead = tempHead.next

    def findRegister(self, name):
        tempHead = self.head
        while tempHead is not None:
            if tempHead.name == name:
                return tempHead
            tempHead = tempHead.next
        return None


class returnManager(Manager):
    def __init__(self) -> None:
        super().__init__(2)
        self.head = Register(in_name="v0", in_manager=self)
        self.tail = Register(in_prev=self.head, in_name="v1", in_manager=self)
        self.head.next = self.tail

    def LRU(self, in_object):
        # check if the registers are in use
        # if v0 not in use, replace object of register v0 with in_object
        # if v1 not in use, replace object of register v1 with in_object
        if self.search(in_object) is not None:
            return
        tempHead = self.head
        free = False
        while tempHead is not None:
            if tempHead.used is False:
                tempHead.update(in_object)
                self.shuffle(tempHead)
                free = True
                break
            tempHead = tempHead.next
        if not free:
            self.head.clear()
            self.head.update(in_object)
            self.shuffle()

    def LRU_delete(self, register_name: str):
        # delete the register with the name register_name and move the register to the tail
        tempHead = self.head
        while tempHead is not None:
            if tempHead.name == register_name:
                tempHead.clear()
                break
            tempHead = tempHead.next
        # move the tempHead to tail of the list
        if tempHead is not None:
            if tempHead.next is not None:
                tempHead.next.prev = tempHead.prev
            else:
                self.tail = tempHead.prev
            if tempHead.prev is not None:
                tempHead.prev.next = tempHead.next
            else:
                self.head = tempHead.next
            self.tail.next = tempHead
            tempHead.prev = self.tail
            tempHead.next = None
            self.tail = tempHead

    def clear(self):
        self.head = Register(in_name="v0", in_manager=self)
        self.tail = Register(in_prev=self.head, in_name="v1", in_manager=self)
        self.head.next = self.tail


class argumentManager(Manager):
    def __init__(self) -> None:
        super().__init__(4)
        list_objects = [
            Register(in_name="a" + str(i), in_manager=self) for i in range(0, 4)
        ]
        self.head = list_objects[0]
        self.tail = list_objects[-1]
        for i in range(3):
            list_objects[i].next = list_objects[i + 1]
            list_objects[i + 1].prev = list_objects[i]

    def LRU(self, in_object):
        # check if the registers are in use
        # if a0 not in use, replace object of register a0 with in_object
        # if a1 not in use, replace object of register a1 with in_object
        # if a2 not in use, replace object of register a2 with in_object
        # if a3 not in use, replace object of register a3 with in_object
        if self.search(in_object) is not None:
            return
        tempHead = self.head
        free = False
        while tempHead is not None:
            if tempHead.used is False:
                tempHead.update(in_object)
                self.shuffle(tempHead)
                free = True
                break
            tempHead = tempHead.next
        if not free:
            self.head.clear()
            self.head.update(in_object)
            self.shuffle()

    def LRU_delete(self, register_name: str):
        # delete the register with the name register_name and move the register to the tail
        tempHead = self.head
        while tempHead is not None:
            if tempHead.name == register_name:
                tempHead.clear()
                break
            tempHead = tempHead.next
        # move the tempHead to tail of the list
        if tempHead is not None:
            if tempHead.next is not None:
                tempHead.next.prev = tempHead.prev
            else:
                self.tail = tempHead.prev
            if tempHead.prev is not None:
                tempHead.prev.next = tempHead.next
            else:
                self.head = tempHead.next
            self.tail.next = tempHead
            tempHead.prev = self.tail
            tempHead.next = None
            self.tail = tempHead

    def clear(self):
        list_objects = [
            Register(in_name="a" + str(i), in_manager=self) for i in range(0, 4)
        ]
        self.head = list_objects[0]
        self.tail = list_objects[-1]
        for i in range(3):
            list_objects[i].next = list_objects[i + 1]
            list_objects[i + 1].prev = list_objects[i]


class temporaryManager(Manager):
    def __init__(self) -> None:
        super().__init__(8)
        list_objects = [
            Register(in_name="t" + str(i), in_manager=self) for i in range(0, 8)
        ]
        self.head = list_objects[0]
        self.tail = list_objects[7]
        for i in range(0, 7):
            list_objects[i].next = list_objects[i + 1]
        for i in range(1, 8):
            list_objects[i].prev = list_objects[i - 1]

    def LRU(self, in_object):
        # check if the registers are in use
        if self.search(in_object) is not None:
            return
        tempHead = self.head
        free = False
        while tempHead is not None:
            if tempHead.used is False:
                tempHead.update(in_object)
                self.shuffle(tempHead)
                free = True
                break
            tempHead = tempHead.next
        if not free:
            self.head.clear()
            self.head.update(in_object)
            self.shuffle()

    def LRU_delete(self, register_name: str):
        # delete the register with the name register_name and move the register to the tail
        tempHead = self.head
        while tempHead is not None:
            if tempHead.name == register_name:
                tempHead.clear()
                self.shuffle(tempHead)
                break
            tempHead = tempHead.next
        # move the tempHead to tail of the list
        if tempHead is not None:
            if tempHead.next is not None:
                tempHead.next.prev = tempHead.prev
            else:
                self.tail = tempHead.prev
            if tempHead.prev is not None:
                tempHead.prev.next = tempHead.next
            else:
                self.head = tempHead.next
            self.tail.next = tempHead
            tempHead.prev = self.tail
            tempHead.next = None
            self.tail = tempHead

    def clear(self):
        list_objects = [
            Register(in_name="t" + str(i), in_manager=self) for i in range(0, 8)
        ]
        self.head = list_objects[0]
        self.tail = list_objects[7]
        for i in range(0, 7):
            list_objects[i].next = list_objects[i + 1]
        for i in range(1, 8):
            list_objects[i].prev = list_objects[i - 1]


class savedManager(Manager):
    def __init__(self) -> None:
        super().__init__(8)
        list_objects = [
            Register(in_name="s" + str(i), in_manager=self) for i in range(0, 8)
        ]
        self.head = list_objects[0]
        self.tail = list_objects[7]
        for i in range(0, 7):
            list_objects[i].next = list_objects[i + 1]
        for i in range(1, 8):
            list_objects[i].prev = list_objects[i - 1]

    def LRU(self, in_object):
        # check if the registers are in use
        if self.search(in_object) is not None:
            return
        tempHead = self.head
        free = False
        while tempHead is not None:
            if tempHead.used is False:
                tempHead.update(in_object)
                self.shuffle(tempHead)
                free = True
                break
            tempHead = tempHead.next
        if not free:
            self.head.clear()
            self.head.update(in_object)
            self.shuffle()

    def LRU_delete(self, register_name: str):
        # delete the register with the name register_name and move the register to the tail
        tempHead = self.head
        while tempHead is not None:
            if tempHead.name == register_name:
                tempHead.clear()
                break
            tempHead = tempHead.next
        # move the tempHead to tail of the list
        if tempHead is not None:
            if tempHead.next is not None:
                tempHead.next.prev = tempHead.prev
            else:
                self.tail = tempHead.prev
            if tempHead.prev is not None:
                tempHead.prev.next = tempHead.next
            else:
                self.head = tempHead.next
            self.tail.next = tempHead
            tempHead.prev = self.tail
            tempHead.next = None
            self.tail = tempHead

    def clear(self):
        list_objects = [
            Register(in_name="s" + str(i), in_manager=self) for i in range(0, 8)
        ]
        self.head = list_objects[0]
        self.tail = list_objects[7]
        for i in range(0, 7):
            list_objects[i].next = list_objects[i + 1]
        for i in range(1, 8):
            list_objects[i].prev = list_objects[i - 1]


class reservedManager(Manager):
    def __init__(self) -> None:
        super().__init__(4)
        self.head = Register(in_name="k0", in_manager=self)
        self.tail = Register(in_prev=self.head, in_name="k1", in_manager=self)
        self.head.next = self.tail

    def LRU(self, in_object):
        # check if the registers are in use
        # if k0 not in use, replace object of register k0 with in_object
        # if k1 not in use, replace object of register k1 with in_object
        if self.search(in_object) is not None:
            return
        tempHead = self.head
        free = False
        while tempHead is not None:
            if tempHead.used is False:
                tempHead.update(in_object)
                self.shuffle(tempHead)
                free = True
                break
            tempHead = tempHead.next
        if not free:
            self.head.clear()
            self.head.update(in_object)
            self.shuffle()

    def LRU_delete(self, register_name: str):
        # delete the register with the name register_name and move the register to the tail
        tempHead = self.head
        while tempHead is not None:
            if tempHead.name == register_name:
                tempHead.clear()
                break
            tempHead = tempHead.next
        # move the tempHead to tail of the list
        if tempHead is not None:
            if tempHead.next is not None:
                tempHead.next.prev = tempHead.prev
            else:
                self.tail = tempHead.prev
            if tempHead.prev is not None:
                tempHead.prev.next = tempHead.next
            else:
                self.head = tempHead.next
            self.tail.next = tempHead
            tempHead.prev = self.tail
            tempHead.next = None
            self.tail = tempHead

    def clear(self):
        self.head = Register(in_name="k0")
        self.tail = Register(in_prev=self.head, in_name="k1")
        self.head.next = self.tail


class floatManager(Manager):
    def __init__(self) -> None:
        super().__init__(32)
        list_objects = [
            Register(in_name="f" + str(i), in_manager=self) for i in range(0, 32)
        ]
        self.head = list_objects[0]
        self.tail = list_objects[31]
        for i in range(0, 31):
            list_objects[i].next = list_objects[i + 1]
        for i in range(1, 32):
            list_objects[i].prev = list_objects[i - 1]

    def LRU(self, in_object):
        # check if the registers are in use
        if self.search(in_object) is not None:
            return
        tempHead = self.head
        free = False
        while tempHead is not None:
            if tempHead.used is False:
                tempHead.update(in_object)
                self.shuffle(tempHead)
                free = True
                break
            tempHead = tempHead.next
        if not free:
            self.head.clear()
            self.head.update(in_object)
            self.shuffle()

    def LRU_delete(self, register_name: str):
        # delete the register with the name register_name and move the register to the tail
        tempHead = self.head
        while tempHead is not None:
            if tempHead.name == register_name:
                tempHead.clear()
                break
            tempHead = tempHead.next
        # move the tempHead to tail of the list
        if tempHead is not None:
            if tempHead.next is not None:
                tempHead.next.prev = tempHead.prev
            else:
                self.tail = tempHead.prev
            if tempHead.prev is not None:
                tempHead.prev.next = tempHead.next
            else:
                self.head = tempHead.next
            self.tail.next = tempHead
            tempHead.prev = self.tail
            tempHead.next = None
            self.tail = tempHead

    def clear(self):
        list_objects = [
            Register(in_name="f" + str(i), in_manager=self) for i in range(0, 32)
        ]
        self.head = list_objects[0]
        self.tail = list_objects[31]
        for i in range(0, 31):
            list_objects[i].next = list_objects[i + 1]
        for i in range(1, 32):
            list_objects[i].prev = list_objects[i - 1]


class singleManager:
    def __init__(self) -> None:
        self.gp = Register(in_name="gp")
        self.sp = Register(in_name="sp")
        self.fp = Register(in_name="fp")
        self.ra = Register(in_name="ra")
        self.zero = Register(in_name="zero")
        self.at = Register(in_name="at")
        self.lo = Register(in_name="lo")
        self.hi = Register(in_name="hi")

    def clear(self):
        self.gp = Register(in_name="gp")
        self.sp = Register(in_name="sp")
        self.fp = Register(in_name="fp")
        self.ra = Register(in_name="ra")
        self.zero = Register(in_name="zero")
        self.at = Register(in_name="at")
        self.lo = Register(in_name="lo")
        self.hi = Register(in_name="hi")

    def findRegister(self, name):
        if name == "gp":
            return self.gp
        if name == "sp":
            return self.sp
        if name == "fp":
            return self.fp
        if name == "ra":
            return self.ra
        if name == "zero":
            return self.zero
        if name == "at":
            return self.at
        if name == "lo":
            return self.lo
        if name == "hi":
            return self.hi
        return None


class dataManager:
    def __init__(self) -> None:
        self.data: [] = [
            {},
            {},
            {},
            {},
            {},
            {},
        ]  # ASCII, float, word, half-word, byte, space
        self.uninitialized: [] = [[], [], [], []]  # char, float, word, array
        self.index = 0
        self.stackSize = 0


class Registers:
    def __init__(self) -> None:
        self.returnManager = returnManager()
        self.argumentManager = argumentManager()
        self.temporaryManager = temporaryManager()
        self.savedManager = savedManager()
        self.reservedManager = reservedManager()
        self.floatManager = floatManager()
        self.singleManager = singleManager()
        self.globalObjects: dataManager = dataManager()

    def clearAll(self):
        self.returnManager.clear()
        self.argumentManager.clear()
        self.temporaryManager.clear()
        self.savedManager.clear()
        self.reservedManager.clear()
        self.singleManager.clear()
        self.floatManager.clear()

    def search(self, in_object):
        temp = self.returnManager.search(in_object)
        if temp is not None:
            self.returnManager.shuffle(temp)
            return temp.name
        temp = self.argumentManager.search(in_object)
        if temp is not None:
            self.argumentManager.shuffle(temp)
            return temp.name
        temp = self.temporaryManager.search(in_object)
        if temp is not None:
            self.temporaryManager.shuffle(temp)
            return temp.name
        temp = self.savedManager.search(in_object)
        if temp is not None:
            self.savedManager.shuffle(temp)
            return temp.name
        temp = self.reservedManager.search(in_object)
        if temp is not None:
            self.reservedManager.shuffle(temp)
            return temp.name
        temp = self.floatManager.search(in_object)
        if temp is not None:
            self.floatManager.shuffle(temp)
            return temp.name
        return None

    def searchRegister(self, name):
        if name.startswith("f"):
            return self.floatManager.findRegister(name)
        if name.startswith("t"):
            return self.temporaryManager.findRegister(name)
        if name.startswith("a"):
            return self.argumentManager.findRegister(name)
        if name.startswith("s"):
            return self.savedManager.findRegister(name)
        if name.startswith("k"):
            return self.reservedManager.findRegister(name)
        if name.startswith("v"):
            return self.returnManager.findRegister(name)
        return self.singleManager.findRegister(name)

    def shuffle_name(self, register_name: str):
        if register_name.startswith("f"):
            self.floatManager.shuffle_name(register_name)
        elif register_name.startswith("t"):
            self.temporaryManager.shuffle_name(register_name)
        elif register_name.startswith("a"):
            self.argumentManager.shuffle_name(register_name)
        elif register_name.startswith("s"):
            self.savedManager.shuffle_name(register_name)
        elif register_name.startswith("k"):
            self.reservedManager.shuffle_name(register_name)
        elif register_name.startswith("v"):
            self.returnManager.shuffle_name(register_name)


class Register:
    def __init__(
        self,
        in_prev=None,
        in_next=None,
        in_object=None,
        in_register=None,
        in_name=None,
        in_manager=None,
    ) -> None:
        self.name = in_name
        self.prev = in_prev
        self.next = in_next
        self.object = in_object
        self.register = in_register
        self.manager = in_manager
        self.used = False if self.object is None else True

    def update(self, in_object):
        self.object = in_object
        self.object.register = self
        self.used = True

    def clear(self):
        self.object.register = None
        self.object = None
        self.used = False

    def shuffle(self):
        self.manager.shuffle(self)
