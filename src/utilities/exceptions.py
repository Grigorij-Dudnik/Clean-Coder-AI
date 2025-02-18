"""Custom esception messages."""

class MissingEnvironmentVariableError(Exception):
    """Enviromental variable missing."""

class ModuleImportedButNotLocatedError(Exception):
    """A module was imported but the location of its source files is unknown."""
