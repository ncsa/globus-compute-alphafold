module purge
module load cue-login-env/1.0 gcc/11.2.0 ucx/1.11.2 openmpi/4.1.2 cuda/11.6.1 modtree/gpu default

export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION=".75"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"