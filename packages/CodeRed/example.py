from numpy import array, random

from codered.middleware import CodeRedLib

stats_only = True

B = random.randint(
    0, 2, size=(252, 541 * 2), dtype="bool"
)  # Create a random Basis for a [5,16]-code
red = CodeRedLib(B)  # Load it into a fresh CodeRedLib object


def niceprint(B):
    for v in B:
        print("".join(["1" if x else "." for x in v]))
    print()


def stats(B):
    acc = 0
    for v in B.T:
        acc += sum(v)

    print("# of 1: ", acc, "; avg: ", sum(B) / len(B))


if not stats_only:
    stats(red.B)
    niceprint(red.B)  # Print current basis
    niceprint(red.E)  # Print current Epipodal matrix
    print(red.l)  # Print current Profile

    red.LLL()  # Apply LLL

    stats(red.B)
    niceprint(red.B)  # Print current basis
    niceprint(red.E)  # Print current Epipodal matrix
    print(red.l)  # Print current Profile
else:
    stats(red.B)
    print(red.l)  # Print current Profile
    red.LLL()  # Apply LLL
    stats(red.B)
    print(red.l)  # Print current Profile
