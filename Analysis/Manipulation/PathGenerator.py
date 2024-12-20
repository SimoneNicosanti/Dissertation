SEPARATOR = ">"


def generatePath(basePath: str, string: str):
    # return f"{basePath}{SEPARATOR}{string}"
    return f"{string}"


def getBasePath(path: str):
    return path.rsplit(SEPARATOR, 1)[0]
