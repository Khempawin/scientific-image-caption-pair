from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    if rank == 0:
        # Program for rank 0
        print(f"Hello from {rank} : {__name__}")
    else:
        # Program for other rank processes
        print(f"Hello from {rank} : {__name__}")

if __name__ == "__main__":
    main()