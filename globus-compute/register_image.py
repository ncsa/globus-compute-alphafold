from globus_compute_sdk import Client
gcc = Client()

print(gcc.register_container("/u/ritwikd2/alphafold-docker-image_latest.sif", "singularity"))