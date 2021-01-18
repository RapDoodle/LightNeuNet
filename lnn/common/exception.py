
class ModelError(Exception):
    """The exception is raised when an error occurs in the model models"""

    def __init__(self, message):
        super().__init__(message)

class NotImplementedError(Exception):
    """The exception is raised when the user tries to use a section of 
    code that the programmer is too lazy to implement"""

    def __init__(self):
        super().__init__('the function is not implemented')