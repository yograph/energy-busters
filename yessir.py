def bitstrings(zeros, ones, length, bits):
    """
    Recursively generates all bitstrings of length n with k zeros and (n-k) ones.

    :param zeros: number of zeros in the bitstring
    :param ones: number of ones in the bitstring
    :param length: length of the bitstring
    :param bits: array to store the bits of the bitstring
    :return: None
    """
    # Print output is both is already done
    if zeros == 0 and ones == 0:
        print("".join(map(str, bits[:length])))
        return

    # If zero is more than 1, always start with 0
    if zeros > 0:
        bits[length] = 0
        bitstrings(zeros - 1, ones, length + 1, bits)

    # This is when one is greater than zero, our condition is fullfilled
    if ones > zeros:
        bits[length] = 1
        bitstrings(zeros, ones - 1, length + 1, bits)
