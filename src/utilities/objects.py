"""
This file contains different object classes.
"""

class CodeFile:
    def __init__(self, filename, is_modified=False):
        self.filename = filename
        self.is_modified = is_modified

    def __str__(self):
        return self.filename
