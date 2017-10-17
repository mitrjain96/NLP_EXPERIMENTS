def isInt(val):
    try:
        int(val)
        return True
    except ValueError:
        return False
