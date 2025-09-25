try:
    from mpi4py import MPI

    comm_world = MPI.COMM_WORLD
    verbose = comm_world.rank == 0
    comm_size = comm_world.size
except ImportError:
    MPI = None
    comm_world = None
    verbose = True
    comm_size = 1

__all__ = ["comm_world", "verbose", "comm_size"]
