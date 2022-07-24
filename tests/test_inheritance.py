from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AbstractClass(ABC):
    """
    Abstract class for testing.
    """
    # Abstract fields can be inherited, initialized, and whatnot
    test_variable: int = 0

    # Abstract dunder methods can have an implementation and be inherited.
    @abstractmethod
    def __post_init__(self):
        self.test_variable += 1

    # Abstract public methods can have an implementation and be inherited.
    @abstractmethod
    def public_method(self):
        print(f"I am a public Abstract. I contain {self.test_variable}")

    @abstractmethod
    def _protected_method(self):
        print(f"I am a protected Abstract. I contain {self.test_variable}")

    # Private methods can not be inherited, thus can't be abstract
    # @abstractmethod
    # def __private_method(self):
    #     ...


@dataclass
class ParentClass(AbstractClass):
    """
    Parent class for testing.
    """

    def __post_init__(self):
        super().__post_init__()
        self.test_variable += 1

    def public_method(self):
        super().public_method()
        print(f"I am a public Parent. I contain {self.test_variable}.")

    # For implemented abstract methods, the implementation can be inherited
    #   with the super() call, but can still be easily overridden with a whole
    #   new implementation (without having the super() call).
    def _protected_method(self):
        print(f"I am a protected Parent. I contain {self.test_variable}.")

    def __private_method(self):
        print(f"I am a private Parent. I contain {self.test_variable}.")


@dataclass
class ChildClass(ParentClass):
    """
    Child class for testing.
    """

    def __post_init__(self):
        super().__post_init__()
        self.test_variable += 1

    def public_method(self):
        super().public_method()
        print(f"I am a public Child. I contain {self.test_variable}.")

    def _protected_method(self):
        super()._protected_method()
        print(f"I am a protected Child. I contain {self.test_variable}.")

    # Child can have such a method but can't invoke it, not even from within
    # def __private_method(self):
    #     super().__private_method()
    #     print(f"I am a private Child. I contain {self.test_variable}.")

    def invoke_private_method_from_parent(self):
        # Child can't access / invoke private methods of Parent
        # super().__private_method()
        # print(f"I am Child, invoking Parent's private method from a public.")

        print(f"This {self.test_variable} is my (Child's) variable.")
        # Child can't access Parent's fields with super()
        # print(f"This {super().test_variable} is my Parent's variable.")

    def print_super(self):
        print(f"This is my super()man: {super()}")


def test_abstract_class():
    """
    Tests the abstract class.
    """
    parent = ParentClass()
    child = ChildClass()
    assert parent.test_variable == 2
    assert child.test_variable == 3

    print("")

    parent.public_method()
    parent._protected_method()
    # Private methods can't be invoked from outside the Parent class
    # parent.__private_method()

    print("")

    child.public_method()
    child._protected_method()
    # Child methods can't be invoked from outside the Child class
    # child.__private_method()

    print("")

    child.invoke_private_method_from_parent()
    child.print_super()
