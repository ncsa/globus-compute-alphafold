from globus_compute_sdk import Client
gcc = Client()

print(gcc.register_container("/projects/bbmi/alphafold/alphafold-docker-image_common.sif", "singularity"))