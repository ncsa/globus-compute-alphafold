### Run Alphafold using Globus-Compute



Run the alphafold model by setting up a globus-compute endpoint on the NCSA Delta Cluster:

1. Install globus-compute:

```shell
conda create env --name alphafold python=3.9

conda activate alphafold

pip install globus-compute-endpoint==2.2.0
```

2. Create a globus-compute-endpint:

```shell
globus-compute-endpoint configure alphafold
```

3. Modify the config file:
```shell
vi ~/.globus_compute/alphafold/config.py
```

```python
from parsl.addresses import address_by_interface
from parsl.launchers import SrunLauncher
from parsl.providers import SlurmProvider
from globus_compute_endpoint.endpoint.utils.config import Config
from globus_compute_endpoint.executors import HighThroughputExecutor

user_opts = {
  'delta': {
    'worker_init': 'pwd',
    'scheduler_options': '#SBATCH --account=bbmi-delta-gpu --gpus=1 ',
  }
}

config = Config(
  executors=[
    HighThroughputExecutor(
      max_workers_per_node=10,
      address=address_by_interface('hsn0'),
      scheduler_mode='soft',
      worker_mode='singularity_reuse',
      container_type='singularity',
      container_cmd_options="--nv --bind /scratch/bblq/parthpatel7173/alphafold_files/globusComputeTesting/input:/mnt/fasta_path_0 --bind /scratch/bblq/parthpatel7173/alphafold_files/globusComputeTesting/output/:/mnt/output --bind /ime/bblq/park3/data_hyun_official/uniref90:/mnt/uniref90_database_path --bind /ime/bblq/park3/data_hyun_official/mgnify:/mnt/mgnify_database_path --bind /ime/bblq/park3/data_hyun_official:/mnt/data_dir --bind /ime/bblq/park3/data_hyun_official/pdb_mmcif/mmcif_files:/mnt/template_mmcif_dir --bind /ime/bblq/park3/data_hyun_official/pdb_mmcif:/mnt/obsolete_pdbs_path --bind /ime/bblq/park3/data_hyun_official/pdb70:/mnt/pdb70_database_path --bind /ime/bblq/park3/data_hyun_official/uniref30:/mnt/uniref30_database_path --bind /ime/bblq/park3/data_hyun_official/bfd:/mnt/bfd_database_path",
      provider=SlurmProvider(
        partition='gpuA100x4-interactive',
        launcher=SrunLauncher(),
        scheduler_options=user_opts['delta']['scheduler_options'],
        worker_init=user_opts['delta']['worker_init'],
        nodes_per_block=1,
        init_blocks=0,
        min_blocks=0,
        max_blocks=1,
        walltime='00:03:00'
      ),
    )
  ],
)
```



4. Run Alphafold using the following command:

```shell
python3.9 -m venv venv

source venv/bin/activate

pip install globus-compute-sdk==2.2.0

pip install git+https://github.com/ritwik-deshpande/alphafold.git
```

```shell
python3 run_alphafold.py \
--fasta_paths=/mnt/fasta_path_0/mchu.fasta \
--max_template_date=2022-01-01 \
--data_dir=/mnt/data_dir \
--output_dir=<local path where results will be stored> \
--use_gpu_relax=False \
--perform_MD_only=True \
--uniref90_database_path=/mnt/uniref90_database_path/uniref90.fasta \
--mgnify_database_path=/mnt/mgnify_database_path/mgy_clusters_2022_05.fa \
--template_mmcif_dir=/mnt/template_mmcif_dir \
--obsolete_pdbs_path=/mnt/obsolete_pdbs_path/obsolete.dat \
--bfd_database_path=/mnt/bfd_database_path/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
--uniref30_database_path=/mnt/uniref30_database_path/UniRef30_2021_03 \
--pdb70_database_path=/mnt/pdb70_database_path/pdb70
```


