import torch
import socket
from mpi4py import MPI


def setup():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ssize = comm.Get_size()
    hostname = socket.gethostname()
    device_count = torch.cuda.device_count()

    data = {
        'rank': rank,
        'hostname': hostname,
        'ip_addr': socket.gethostbyname(hostname),
        'gpu_count': device_count,
    }
    all_data = comm.allgather(data)

    # Find all the ranks on same host
    common_ranks = [
        ii for ii, obj in enumerate(all_data)
        if obj['hostname'] == hostname
    ]
    if len(common_ranks) != device_count:
        print('Something is missing!', len(common_ranks), device_count, hostname)

    gpu = common_ranks.index(rank)
    master_ip_addr = all_data[0]['ip_addr']
    print(data)

    print('###')
    print(f"{master_ip_addr}\t{ssize}\t{rank}\t{gpu}")


if __name__ == '__main__':
    setup()
