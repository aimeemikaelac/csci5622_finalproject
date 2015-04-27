
def stringToInt(s):
    return int(float(s))

def positionToAccuracy(p):
    # Converts a position to a 1 or -1 indicating question correctness
    # eg.  30 --> 1
    #     -45 --> -1
    return p/abs(p)
