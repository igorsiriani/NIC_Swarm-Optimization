import numpy as np


def read_tsp(input_file):
    """Reads an input .tsp file.
    Args:
        input_file (str): A string holding the input file's path.
    Returns:
        A set of nodes from the loaded file.
    """

    # Opening input file
    f = open(input_file, 'r')

    # While not finding data
    while f.readline().find('NODE_COORD_SECTION') == -1:
        # Keeps reading
        pass

    # Creating an empty list
    nodes = []

    # While finding data is possible
    while True:
        # Reading every line
        data = f.readline()

        # If reaching end of file
        if data.find('EOF') != -1:
            # Break the process
            break

        # Splits the line
        (_, x, y) = data.split()

        # Convert (x, y) pairs into float
        x = float(x)
        y = float(y)

        # Appends to list
        nodes.append(np.array([x, y]))

    return nodes


def create_distances(nodes):
    """Calculates the euclidean distance between the nodes.
    Args:
        nodes (list): A list of (x, y) nodes' pairs.
    Returns:
        A distance matrix from input nodes.
    """

    # Gathering the amount of nodes
    size = len(nodes)

    # Creating an empty distance matrix
    distances = np.zeros((size, size))

    # For every pair
    for i in range(size):
        # For every pair
        for j in range(size):
            # If pairs are different, calculate the distance
            distances[i][j] = np.linalg.norm(nodes[i] - nodes[j])

            # If pairs are equal
            if i == j:
                # Apply an infinite distance
                distances[i][j] = np.inf

    return distances