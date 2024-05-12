# causality.py
# Toolkit for parsing and working with causality graphs for specification mining.
# Written by Ghali Omar Boutaib <ghaliomar@usf.edu>
# SEES Lab, College of Engineering, University of South Florida

from dataclasses import dataclass
from enum import Enum
from pprint import pprint


class Direction(Enum):
    """Defines the direction of a message. Either a REQuest or a RESPonse."""

    REQ, RESP = range(2)


class Position(Enum):
    """Defines the position of a given message in a sequence. Either INITIAL, INTERMEDIARY, or FINAL."""

    INITIAL, INTERMEDIARY, FINAL = range(3)


@dataclass
class Message:
    """Defines a message."""

    id: int
    origin: str
    destination: str
    operation: str
    direction: Direction
    position: Position


def causal(a: Message, b: Message) -> bool:
    """Determines if a message 'a' can cause a message 'b'

    Args:
        a (Message): Preceding message
        b (Message): Next message

    Returns:
        bool: True if message 'a' can cause message 'b', False otherwise
    """

    # Instead of defining the terms of success, defines each possible failure of the function.
    # Inverts the whole expression to produce expected output.
    # Why? Because it's slightly easier like this.
    return not (
        (a.direction == Direction.RESP and b.direction == Direction.REQ)
        or (a.destination != b.origin)
        or (a.position == Position.FINAL)
        or (b.position == Position.INITIAL)
        or (a.id == b.id)
        # Below case breaks if opcode is different throughout.
        # Only uncomment in the specific situation where you need all possible sequences to only have one opcode.
        # or (a.operation != b.operation)
    )


def build_relationships(graph: dict[int, set[int]], library: dict[int, Message]):
    """Builds all causal relationships in a given graph

    Args:
        graph (dict[int, set[int]]): Given graph
        library (dict[int, Message]): Given library linking ids to Message data
    """

    for a_id in graph:
        for b_id in graph:
            if causal(library[a_id], library[b_id]):
                graph[a_id].add(b_id)


def parse_direction(original: str) -> Direction:
    """Parses a direction string into a Direction object

    Args:
        original (str): A direction string from causality.txt, must be either "req" or "resp"

    Raises:
        ValueError: Can be raised if the value is anything else than "req" or "resp"

    Returns:
        Direction: Resulting Direction object
    """

    if original == "req":
        return Direction.REQ
    elif original == "resp":
        return Direction.RESP
    else:
        raise ValueError(
            "Invalid message direction provided. Expected either 'req' or 'resp'."
        )


def recursive_build_sequences(
    starting_id: int,
    accumulated: list[int],
    sequence_destination: list[list[int]],
    max_depth: int,
    graph: dict[int, set[int]],
    library: dict[int, Message],
):
    """Recursively builds possible message sequences starting at a given message.

    Args:
        starting_id (int): Id number of the starting message
        accumulated (list[int]): A list/array holding the current accumulated message sequence
        sequence_destination (list[list[int]]): Destination list/array for successfully build sequences
        max_depth (int): Maximum depth of a built sequence
        graph (dict[int, set[int]]): Given graph
        library (dict[int, Message]): Given library linking ids to Message data
    """

    # Rule out cyclic cases
    if starting_id in accumulated:
        return

    new_accumulation: list[int] = accumulated + [starting_id]

    if len(new_accumulation) > max_depth:
        return

    if library[starting_id].position == Position.FINAL:
        sequence_destination.append(new_accumulation)
        return

    for next_id in graph[starting_id]:
        recursive_build_sequences(
            next_id, new_accumulation, sequence_destination, max_depth, graph, library
        )


def build_sequences(
    starting_id: int,
    max_depth: int,
    graph: dict[int, set[int]],
    library: dict[int, Message],
) -> list[list[int]]:
    """Simplifies the use of recursive_build_sequences and directly returns the value instead of relying on an external state variable

    Args:
        starting_id (int): Id number of the starting message
        max_depth (int): Maximum depth of the sequence
        graph (dict[int, set[int]]): Given graph
        library (dict[int, Message]): Given library linking ids to Message data

    Returns:
        list[list[int]]: All resulting possible message sequences
    """

    dest = []

    recursive_build_sequences(starting_id, [], dest, max_depth, graph, library)

    return dest


def project_sequences(
    sequences: list[list[int]], whitelist: set[int]
) -> list[list[int]]:
    """Given generated message sequences, filters out/projects for a given whitelist of message ids. Conserves relative positioning in each message sequence and discards empty message sequences.

    Args:
        sequences (list[list[int]]): Given sequences to project/filter
        whitelist (set[int]): Whitelist of message ids to project for

    Returns:
        list[list[int]]: Resulting filtered/projected sequences
    """

    new_sequences = []

    for sequence in sequences:
        new_sequence = []

        for id in sequence:
            if id in whitelist:
                new_sequence.append(id)

        if len(new_sequence) > 1 and not new_sequence in new_sequences:
            new_sequences.append(new_sequence)

    new_sequences.sort(key=len)

    return new_sequences


def pair_filter(sequence: list[int], a: int, b: int) -> bool:
    """Assesses if a given message sequence fulfills a pair requirement. i.e. if either or both messages from the specified pair exist in the sequence, they must be neighbors in the specified order.

    Args:
        sequence (list[int]): Message sequence
        a (int): First message id of the pair
        b (int): Second message id of the pair

    Returns:
        bool: Logical test result
    """
    if a in sequence and b in sequence:
        return sequence.index(a) + 1 == sequence.index(b)
    elif a in sequence or b in sequence:
        return False
    else:
        return True


def build_graph(
    graph_file: str = "causality.txt",
) -> tuple[dict[int, Message], dict[int, set[int]]]:
    """Parses a given causality graph txt file into a library and a graph with no relationships. Run build_relationships on the graph to build the relationships between the nodes.

    Args:
        graph_file (str, optional): Name of the causality graph txt file. Defaults to "causality.txt".

    Returns:
        tuple[dict[int, Message], dict[int, set[int]]]: library, graph
    """

    library: dict[int, Message] = {}

    with open(graph_file, "r") as file:
        sections = list(
            map(
                lambda section: list(
                    map(
                        lambda line: list(
                            map(lambda piece: piece.strip(), line.split(":"))
                        ),
                        filter(lambda line: line, section.split("\n")),
                    )
                ),
                file.read().split("#"),
            )
        )

        section_positions: list[Position] = [
            Position.INITIAL,
            Position.INTERMEDIARY,
            Position.FINAL,
        ]

        for section_index, section in enumerate(sections):
            for message in section:
                library[int(message[0])] = Message(
                    int(message[0]),
                    message[1],
                    message[2],
                    message[3],
                    parse_direction(message[4]),
                    section_positions[section_index],
                )

    graph: dict[int, set[int]] = {}

    for message_id in library:
        graph[message_id] = set()

    build_relationships(graph, library)

    return (library, graph)


def generate_all_sequences(
    library: dict[int, Message], graph: dict[int, set[int]], max_depth: int
) -> list[list[int]]:
    """Generates all possible sequences given a causality graph

    Args:
        library (dict[int, Message]): Message library
        graph (dict[int, set[int]]): Prebuilt causality graph
        max_depth (int): Maximum depth of any given sequence

    Returns:
        list[list[int]]: Resulting sequences
    """

    sequences: list[list[int]] = []

    for message_id in library:
        if library[message_id].position == Position.INITIAL:
            sequences += build_sequences(message_id, max_depth, graph, library)

    return sequences
