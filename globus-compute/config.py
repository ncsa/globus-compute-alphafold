from parsl.addresses import address_by_interface
from parsl.launchers import SrunLauncher
from parsl.providers import SlurmProvider
from globus_compute_endpoint.endpoint.utils.config import Config
from globus_compute_endpoint.executors import HighThroughputExecutor

user_opts = {
  'delta': {
    'worker_init': 'bash /u/ritwikd2/worker_init.sh',
    'scheduler_options': '#SBATCH --account=bbmi-delta-gpu --mem=100g --nodes=1 --gpus-per-node=4',
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
        partition='gpuA40x4',
        launcher=SrunLauncher(),
        scheduler_options=user_opts['delta']['scheduler_options'],
        worker_init=user_opts['delta']['worker_init'],
        nodes_per_block=1,
        init_blocks=0,
        min_blocks=0,
        max_blocks=1,
        walltime='00:30:00'
      ),
    )
  ],
)