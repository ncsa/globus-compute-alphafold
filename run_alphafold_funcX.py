import enum
import json
import os
import pathlib
import random
import sys
import time
import functools
import datetime

from absl import app
from absl import flags
from absl import logging

from concurrent.futures import as_completed
from globus_compute_sdk import Executor

logging.set_verbosity(logging.INFO)

ENDPOINT_ID = "070e6cc9-60a9-44ec-be40-e54d81a1ed69"
CONTAINER_ID = "3fb0c84f-c8a9-461f-9d9a-faa1530b39a5"

@enum.unique
class ModelsToRelax(enum.Enum):
    ALL = 0
    BEST = 1
    NONE = 2


MODEL_PRESETS = {
    'monomer': (
        'model_1',
        'model_2',
        'model_3',
        'model_4',
        'model_5',
    ),
    'monomer_ptm': (
        'model_1_ptm',
        'model_2_ptm',
        'model_3_ptm',
        'model_4_ptm',
        'model_5_ptm',
    ),
    'multimer': (
        'model_1_multimer_v3',
        'model_2_multimer_v3',
        'model_3_multimer_v3',
        'model_4_multimer_v3',
        'model_5_multimer_v3',
    ),
}
MODEL_PRESETS['monomer_casp14'] = MODEL_PRESETS['monomer']

flags.DEFINE_list('fasta_paths', None, 'Paths to FASTA files, each containing a prediction '
                                       'target that will be folded one after another. If a FASTA file contains '
                                       'multiple sequences, then it will be folded as a multimer. Paths should be '
                                       'separated by commas. All FASTA paths must have a unique basename as the '
                                       'basename is used to name the output directories for each prediction.')

flags.DEFINE_string('data_dir', '/mnt/data_dir', 'Path to directory of supporting data.')

flags.DEFINE_string('output_dir', None, 'Path to a directory that will store the results.')

flags.DEFINE_string('uniref90_database_path', '/mnt/uniref90_database_path/uniref90.fasta', 'Path to the Uniref90 '
                                                                                            'database for use by JackHMMER.')
flags.DEFINE_string('mgnify_database_path', '/mnt/mgnify_database_path/mgy_clusters_2022_05.fa', 'Path to the MGnify '
                                                                                                 'database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path',
                    '/mnt/bfd_database_path/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt',
                    'Path to the BFD '
                    'database for use by HHblits.')
flags.DEFINE_string('small_bfd_database_path', None, 'Path to the small '
                                                     'version of BFD used with the "reduced_dbs" preset.')
flags.DEFINE_string('uniref30_database_path', '/mnt/uniref30_database_path/UniRef30_2021_03', 'Path to the UniRef30 '
                                                                                              'database for use by HHblits.')
flags.DEFINE_string('uniprot_database_path', None, 'Path to the Uniprot '
                                                   'database for use by JackHMMer.')
flags.DEFINE_string('pdb70_database_path', '/mnt/pdb70_database_path/pdb70', 'Path to the PDB70 '
                                                                             'database for use by HHsearch.')
flags.DEFINE_string('pdb_seqres_database_path', None, 'Path to the PDB '
                                                      'seqres database for use by hmmsearch.')
flags.DEFINE_string('template_mmcif_dir', '/mnt/template_mmcif_dir', 'Path to a directory with '
                                                                     'template mmCIF structures, each named <pdb_id>.cif')
flags.DEFINE_string('max_template_date', '2022-01-01', 'Maximum template release date '
                                                       'to consider. Important if folding historical test sets.')
flags.DEFINE_string('obsolete_pdbs_path', '/mnt/obsolete_pdbs_path/obsolete.dat', 'Path to file containing a '
                                                                                  'mapping from obsolete PDB IDs to the PDB IDs of their '
                                                                                  'replacements.')
flags.DEFINE_enum('db_preset', 'full_dbs',
                  ['full_dbs', 'reduced_dbs'],
                  'Choose preset MSA database configuration - '
                  'smaller genetic database config (reduced_dbs) or '
                  'full genetic database config  (full_dbs)')

flags.DEFINE_enum('model_preset', 'monomer',
                  ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'],
                  'Choose preset model configuration - the monomer model, '
                  'the monomer model with extra ensembling, monomer model with '
                  'pTM head, or multimer model')

flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                                         'to obtain a timing that excludes the compilation time, '
                                         'which should be more indicative of the time required for '
                                         'inferencing many proteins.')

flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                                          'pipeline. By default, this is randomly generated. Note '
                                          'that even if this is set, Alphafold may still not be '
                                          'deterministic, because processes like GPU inference are '
                                          'nondeterministic.')

flags.DEFINE_integer('num_multimer_predictions_per_model', 5, 'How many '
                                                              'predictions (each with a different random seed) will be '
                                                              'generated per model. E.g. if this is 2 and there are 5 '
                                                              'models then there will be 10 predictions per input. '
                                                              'Note: this FLAG only applies if model_preset=multimer')

flags.DEFINE_boolean('use_precomputed_msas', False, 'Whether to read MSAs that '
                                                    'have been written to disk instead of running the MSA '
                                                    'tools. The MSA files are looked up in the output '
                                                    'directory, so it must stay the same between multiple '
                                                    'runs that are to reuse the MSAs. WARNING: This will not '
                                                    'check if the sequence, database or configuration have '
                                                    'changed.')

flags.DEFINE_enum_class('models_to_relax', ModelsToRelax.BEST, ModelsToRelax,
                        'The models to run the final relaxation step on. '
                        'If `all`, all models are relaxed, which may be time '
                        'consuming. If `best`, only the most confident model '
                        'is relaxed. If `none`, relaxation is not run. Turning '
                        'off relaxation might result in predictions with '
                        'distracting stereochemical violations but might help '
                        'in case you are having issues with the relaxation '
                        'stage.')

flags.DEFINE_boolean('use_gpu_relax', None, 'Whether to relax on GPU. '
                                            'Relax on GPU can be much faster than CPU, so it is '
                                            'recommended to enable if possible. GPUs must be available'
                                            ' if this setting is enabled.')
flags.DEFINE_boolean('perform_MD_only', None, 'Whether to use MD only..')

flags.DEFINE_boolean('use_amber', None, 'Whether to use Amber as MD engine or CHARMM')

# Colabfold
# https://github.com/sokrypton/ColabFold/blob/main/colabfold/alphafold/models.py#L10
flags.DEFINE_integer('num_recycles', None, 'Different model_preset has different num_recycles...')
flags.DEFINE_float('recycle_early_stop_tolerance', None, 'Only multimer has this option set...')
flags.DEFINE_integer('num_ensemble', 1, '1 is a default while CASP is 8...')
flags.DEFINE_integer('max_seq', None, 'Number of cluster centers?')
flags.DEFINE_integer('max_extra_seq', None, 'Number of cluster centers?')
flags.DEFINE_boolean('use_fuse', False, 'Global config for mono and multimer... ')
flags.DEFINE_boolean('use_bfloat16', True, 'Only for multimer')
flags.DEFINE_boolean('use_dropout', False, 'Global config for mono and multimer...')

FLAGS = flags.FLAGS


def _check_flag(flag_name: str,
                other_flag_name: str,
                should_be_set: bool):
    if should_be_set != bool(FLAGS[flag_name].value):
        verb = 'be' if should_be_set else 'not be'
        raise ValueError(f'{flag_name} must {verb} set when running with '
                         f'"--{other_flag_name}={FLAGS[other_flag_name].value}".')


def run_one_openmm(model_name, relax_metrics, unrelaxed_proteins, timings):
    import time
    import datetime
    from alphafold.relax import relax

    t_0 = time.time()

    amber_relaxer = relax.AmberRelaxation(
        max_iterations=0,
        tolerance=2.39,
        stiffness=10.0,
        exclude_residues=[],
        max_outer_iterations=3,
        use_gpu=False,
        use_amber=False)

    relaxed_pdb_str, _, violations = amber_relaxer.process(prot=unrelaxed_proteins[model_name])
    relax_metrics[model_name] = {
        'remaining_violations': violations,
        'remaining_violations_count': sum(violations)
    }
    end_time = time.time()
    t_diff = end_time - t_0
    timings[f'relax_{model_name}'] = t_diff
    timings[f'relax_{model_name}_start_time'] = datetime.datetime.fromtimestamp(t_0).strftime("%H:%M:%S")
    timings[f'relax_{model_name}_end_time'] = datetime.datetime.fromtimestamp(end_time).strftime("%H:%M:%S")

    return relaxed_pdb_str, relax_metrics, timings


def create_features_dict(
        output_dir,
        pdb_seqres_database_path,
        template_mmcif_dir,
        max_template_date,
        obsolete_pdbs_path,
        pdb70_database_path,
        uniref90_database_path,
        mgnify_database_path,
        bfd_database_path,
        uniref30_database_path,
        small_bfd_database_path,
        use_precomputed_msas,
        uniprot_database_path,
        run_multimer_system,
        fasta_path,
):
    import os
    import pickle
    import shutil
    from alphafold.data import pipeline
    from alphafold.data import pipeline_multimer
    from alphafold.data import templates
    from alphafold.data.tools import hhsearch
    from alphafold.data.tools import hmmsearch

    jackhmmer_binary_path = shutil.which('jackhmmer')
    hhblits_binary_path = shutil.which('hhblits')
    hhsearch_binary_path = shutil.which('hhsearch')
    hmmsearch_binary_path = shutil.which('hmmsearch')
    hmmbuild_binary_path = shutil.which('hmmbuild')
    kalign_binary_path = shutil.which('kalign')

    if run_multimer_system:
        template_searcher = hmmsearch.Hmmsearch(
            binary_path=hmmsearch_binary_path,
            hmmbuild_binary_path=hmmbuild_binary_path,
            database_path=pdb_seqres_database_path)

        template_featurizer = templates.HmmsearchHitFeaturizer(
            mmcif_dir=template_mmcif_dir,
            max_template_date=max_template_date,
            max_hits=20,
            kalign_binary_path=kalign_binary_path,
            release_dates_path=None,
            obsolete_pdbs_path=obsolete_pdbs_path)
    else:
        template_searcher = hhsearch.HHSearch(
            binary_path=hhsearch_binary_path,
            databases=[pdb70_database_path])
        template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=template_mmcif_dir,
            max_template_date=max_template_date,
            max_hits=20,
            kalign_binary_path=kalign_binary_path,
            release_dates_path=None,
            obsolete_pdbs_path=obsolete_pdbs_path)

    monomer_data_pipeline = pipeline.DataPipeline(
        jackhmmer_binary_path=jackhmmer_binary_path,
        hhblits_binary_path=hhblits_binary_path,
        uniref90_database_path=uniref90_database_path,
        mgnify_database_path=mgnify_database_path,
        bfd_database_path=bfd_database_path,
        uniref30_database_path=uniref30_database_path,
        small_bfd_database_path=small_bfd_database_path,
        template_searcher=template_searcher,
        template_featurizer=template_featurizer,
        use_small_bfd=False,
        use_precomputed_msas=use_precomputed_msas)

    if run_multimer_system:
        data_pipeline = pipeline_multimer.DataPipeline(
            monomer_data_pipeline=monomer_data_pipeline,
            jackhmmer_binary_path=jackhmmer_binary_path,
            uniprot_database_path=uniprot_database_path,
            use_precomputed_msas=use_precomputed_msas)
    else:
        data_pipeline = monomer_data_pipeline

    features_output_path = os.path.join(output_dir, 'features.pkl')


    if not os.path.exists(features_output_path):
        msa_output_dir = os.path.join(output_dir, 'msas')
        if not os.path.exists(msa_output_dir):
            os.makedirs(msa_output_dir)
        feature_dict = data_pipeline.process(
            input_fasta_path=fasta_path,
            msa_output_dir=msa_output_dir)

        with open(features_output_path, 'wb') as f:
            pickle.dump(feature_dict, f, protocol=4)

        return "Created new feature_dict file"
    else:
        return "feature_dict file already present"

def predict_one_structure(
        output_dir,
        num_recycles,
        recycle_early_stop_tolerance,
        num_ensemble,
        max_seq,
        max_extra_seq,
        use_bfloat16,
        use_dropout,
        model_preset,
        data_dir,
        model_name,
        model_random_seed,
        timings,
        ranking_confidences,
        unrelaxed_pdbs,
        unrelaxed_proteins):
    import time
    import datetime
    import os
    import pickle
    import jax.numpy as jnp
    import numpy as np
    from typing import Any, Dict
    from alphafold.model import data
    from alphafold.model import model
    from alphafold.model import edit_config
    from alphafold.common import confidence
    from alphafold.common import protein
    from alphafold.model import config
    from alphafold.common import residue_constants

    def _jnp_to_np(output: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively changes jax arrays to numpy arrays."""
        for k, v in output.items():
            if isinstance(v, dict):
                output[k] = _jnp_to_np(v)
            elif isinstance(v, jnp.ndarray):
                output[k] = np.array(v)
        return output

    def _np_to_jnp(output: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively changes numpy arrays to jax arrays."""
        for k, v in output.items():
            if isinstance(v, dict):
                output[k] = _np_to_jnp(v)
            elif isinstance(v, np.ndarray):
                output[k] = jnp.array(v)
        return output

    def _save_confidence_json_file(
            plddt: np.ndarray, output_dir: str, model_name: str
    ) -> None:
        confidence_json = confidence.confidence_json(plddt)

        # Save the confidence json.
        confidence_json_output_path = os.path.join(
            output_dir, f'confidence_{model_name}.json'
        )
        with open(confidence_json_output_path, 'w') as f:
            f.write(confidence_json)

    def _save_pae_json_file(
            pae: np.ndarray, max_pae: float, output_dir: str, model_name: str
    ) -> None:
        """Check prediction result for PAE data and save to a JSON file if present.

        Args:
          pae: The n_res x n_res PAE array.
          max_pae: The maximum possible PAE value.
          output_dir: Directory to which files are saved.
          model_name: Name of a model.
        """
        pae_json = confidence.pae_json(pae, max_pae)

        # Save the PAE json.
        pae_json_output_path = os.path.join(output_dir, f'pae_{model_name}.json')
        with open(pae_json_output_path, 'w') as f:
            f.write(pae_json)

    print("Data pipeline process started")
    features_output_path = os.path.join(output_dir, 'features.pkl')
    feature_dict = pickle.load(open(features_output_path, 'rb'))


    print(f"Running model {model_name}")
    t_0 = time.time()
    model_config = config.model_config(model_name)
    model_config = edit_config.load_models_config(config=model_config,
                                                  num_recycles=num_recycles,
                                                  recycle_early_stop_tolerance=recycle_early_stop_tolerance,
                                                  num_ensemble=num_ensemble,
                                                  model_order=None,
                                                  max_seq=max_seq,
                                                  max_extra_seq=max_extra_seq,
                                                  use_bfloat16=use_bfloat16,
                                                  use_dropout=use_dropout,
                                                  save_all=True,
                                                  model_suffix=model_preset)

    model_params = data.get_model_haiku_params(model_name=model_name, data_dir=data_dir)

    model_runner = model.RunModel(model_config, model_params)
    processed_feature_dict = model_runner.process_features(feature_dict, random_seed=model_random_seed)
    end_time = time.time()

    timings[f'process_features_{model_name}'] = end_time - t_0
    timings[f'process_features_{model_name}_start_time'] = datetime.datetime.fromtimestamp(t_0).strftime("%H:%M:%S")
    timings[f'process_features_{model_name}_end_time'] = datetime.datetime.fromtimestamp(end_time).strftime("%H:%M:%S")

    t_0 = time.time()
    prediction_result = model_runner.predict(processed_feature_dict, random_seed=model_random_seed)

    end_time = time.time()
    t_diff = end_time - t_0
    timings[f'predict_and_compile_{model_name}'] = t_diff
    timings[f'predict_and_compile_{model_name}_start_time'] = datetime.datetime.fromtimestamp(t_0).strftime("%H:%M:%S")
    timings[f'predict_and_compile_{model_name}_end_time'] = datetime.datetime.fromtimestamp(end_time).strftime(
        "%H:%M:%S")

    print(f"Total JAX model {model_name}  predict time (includes compilation time, see --benchmark): {t_diff}")

    plddt = prediction_result['plddt']
    _save_confidence_json_file(plddt, output_dir, model_name)
    ranking_confidences[model_name] = prediction_result['ranking_confidence']

    if 'predicted_aligned_error' in prediction_result and 'max_predicted_aligned_error' in prediction_result:
        pae = prediction_result['predicted_aligned_error']
        max_pae = prediction_result['max_predicted_aligned_error']
        _save_pae_json_file(pae, float(max_pae), output_dir, model_name)

    # Remove jax dependency from results.
    np_prediction_result = _jnp_to_np(dict(prediction_result))
    # Get a label early on!
    label = 'iptm+ptm' if 'iptm' in prediction_result else 'plddts'

    # Save the model outputs.
    result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
    with open(result_output_path, 'wb') as f:
        pickle.dump(np_prediction_result, f, protocol=4)

    # Add the predicted LDDT in the b-factor column.
    # Note that higher predicted LDDT value means higher model confidence.
    plddt_b_factors = np.repeat(plddt[:, None], residue_constants.atom_type_num, axis=-1)

    unrelaxed_protein = protein.from_prediction(features=processed_feature_dict,
                                                result=prediction_result,
                                                b_factors=plddt_b_factors,
                                                remove_leading_feature_dimension=True)

    unrelaxed_proteins[model_name] = unrelaxed_protein
    unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
    unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')

    with open(unrelaxed_pdb_path, 'w') as f:
        f.write(unrelaxed_pdbs[model_name])

    return timings, unrelaxed_proteins, unrelaxed_pdbs, ranking_confidences, label


def collate_dictionary(dic0, dic1):
    dic0.update(dic1)
    return dic0


def predict_structure(
        fasta_path: str,
        fasta_name: str,
        output_dir: str,
        model_names: list,
        random_seed: int):
    """Predicts structure using AlphaFold for the given sequence."""
    logging.info('Predicting %s', fasta_name)
    timings = {}

    unrelaxed_proteins = {}
    ranking_confidences = {}
    unrelaxed_pdbs = {}

    outputs = []
    futures = []

    num_models = len(model_names)
    with Executor(endpoint_id=ENDPOINT_ID,
                  container_id=CONTAINER_ID) as ex:
        for model_index, model_name in enumerate(model_names):
            futures.append(ex.submit(predict_one_structure,
                                     os.path.join("/mnt/output/", fasta_name),
                                     FLAGS.num_recycles,
                                     FLAGS.recycle_early_stop_tolerance,
                                     FLAGS.num_ensemble,
                                     FLAGS.max_seq,
                                     FLAGS.max_extra_seq,
                                     FLAGS.use_bfloat16,
                                     FLAGS.use_dropout,
                                     FLAGS.model_preset,
                                     FLAGS.data_dir,
                                     model_name,
                                     model_index + random_seed * num_models,
                                     timings,
                                     ranking_confidences,
                                     unrelaxed_pdbs,
                                     unrelaxed_proteins))

    for future in as_completed(futures):
        outputs.append(future.result())

    outputs_ = list(zip(*outputs))
    outputs = outputs_[:4]
    label = outputs_[-1][0]
    assert len(outputs) == 4, "There should be only four tuples ready..."

    timings, unrelaxed_proteins, unrelaxed_pdbs, ranking_confidences = [functools.reduce(collate_dictionary, out) for
                                                                        out in outputs]

    # timings has to be saved to resume MD
    timings_output_path = os.path.join(output_dir, 'timings.json')
    with open(timings_output_path, 'w') as f:
        f.write(json.dumps(timings, indent=4))

    # ranking_confidences has to be saved to resume MD
    ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
    with open(ranking_output_path, 'w') as f:
        f.write(json.dumps({label: ranking_confidences}, indent=4))

    return timings, unrelaxed_proteins, unrelaxed_pdbs, ranking_confidences, label


def structure_ranker(output_dir: str,
                     fasta_name: str,
                     models_to_relax: ModelsToRelax,
                     timings,
                     unrelaxed_proteins,
                     unrelaxed_pdbs: dict,
                     ranking_confidences: dict,
                     label: str,
                     use_amber: bool):
    relaxed_pdbs = {}
    relax_metrics = {}
    # Rank by model confidence.
    ranked_order = [
        model_name for model_name, confidence in
        sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)]

    # Relax predictions.
    if models_to_relax == ModelsToRelax.BEST:
        to_relax = [ranked_order[0]]
    elif models_to_relax == ModelsToRelax.ALL:
        to_relax = ranked_order
    else:
        to_relax = []

    futures = []
    outputs = []
    with Executor(endpoint_id=ENDPOINT_ID,
                  container_id=CONTAINER_ID) as ex:
        for model_name in to_relax:
            futures.append(ex.submit(run_one_openmm, model_name, relax_metrics, unrelaxed_proteins, timings))

    for future in as_completed(futures):
        outputs.append(future.result())

    outputs = list(zip(*outputs))
    relaxed_pdb_strs = outputs[0]
    relax_metrics, timings = [functools.reduce(collate_dictionary, out) for out in outputs[1:]]

    for model_name, relaxed_pdb_str in zip(to_relax, relaxed_pdb_strs):
        relaxed_pdbs[model_name] = relaxed_pdb_str
        relaxed_output_path = os.path.join(
            output_dir, f'relaxed_{model_name}.pdb')
        with open(relaxed_output_path, 'w') as f:
            f.write(relaxed_pdb_str)

    # Write out relaxed PDBs in rank order.
    for idx, model_name in enumerate(ranked_order):
        suffix = "amber" if use_amber else "charmm"
        ranked_output_path = os.path.join(output_dir, f'ranked_{idx}_{suffix}.pdb')
        with open(ranked_output_path, 'w') as f:
            if model_name in relaxed_pdbs:
                f.write(relaxed_pdbs[model_name])
            else:
                f.write(unrelaxed_pdbs[model_name])

    ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
    with open(ranking_output_path, 'w') as f:
        f.write(json.dumps(
            {label: ranking_confidences, 'order': ranked_order}, indent=4))

    logging.info('Final timings for %s: %s', fasta_name, timings)

    timings_output_path = os.path.join(output_dir, 'timings.json')
    with open(timings_output_path, 'w') as f:
        f.write(json.dumps(timings, indent=4))

    if models_to_relax != ModelsToRelax.NONE:
        relax_metrics_path = os.path.join(output_dir, 'relax_metrics.json')
        with open(relax_metrics_path, 'w') as f:
            f.write(json.dumps(relax_metrics, indent=4))


def main(argv):
    logging.info("\n\n Alphafold Prediction Started")

    total_time = {}
    t_setup_start = time.time()

    use_small_bfd = FLAGS.db_preset == 'reduced_dbs'
    _check_flag('small_bfd_database_path', 'db_preset',
                should_be_set=use_small_bfd)
    _check_flag('bfd_database_path', 'db_preset',
                should_be_set=not use_small_bfd)
    _check_flag('uniref30_database_path', 'db_preset',
                should_be_set=not use_small_bfd)

    run_multimer_system = 'multimer' in FLAGS.model_preset
    _check_flag('pdb70_database_path', 'model_preset',
                should_be_set=not run_multimer_system)
    _check_flag('pdb_seqres_database_path', 'model_preset',
                should_be_set=run_multimer_system)
    _check_flag('uniprot_database_path', 'model_preset',
                should_be_set=run_multimer_system)

    # Check for duplicate FASTA file names.
    fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
    if len(fasta_names) != len(set(fasta_names)):
        raise ValueError('All FASTA paths must have a unique basename.')

    model_names = MODEL_PRESETS[FLAGS.model_preset]

    random_seed = FLAGS.random_seed
    if random_seed is None:
        random_seed = random.randrange(sys.maxsize // len(model_names))
    logging.info('Using random seed %d for the data pipeline', random_seed)

    t_setup_end = time.time()
    total_time['setting_up_time'] = t_setup_end - t_setup_start
    total_time['setting_up_start_time'] = datetime.datetime.fromtimestamp(t_setup_start).strftime("%H:%M:%S")
    total_time['setting_up_end_time'] = datetime.datetime.fromtimestamp(t_setup_end).strftime("%H:%M:%S")

    # Predict structure for each of the sequences.
    t_pred_start = time.time()
    for i, fasta_path in enumerate(FLAGS.fasta_paths):
        additional_timings = {}
        t_0 = time.time()
        fasta_name = fasta_names[i]

        output_dir = os.path.join(FLAGS.output_dir, fasta_name)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Check is features dict is present for those proteins, else creates the dictionary
        with Executor(endpoint_id=ENDPOINT_ID,
                      container_id=CONTAINER_ID) as ex:
            fut = ex.submit(create_features_dict,
                            os.path.join("/mnt/output/", fasta_name),
                            FLAGS.pdb_seqres_database_path,
                            FLAGS.template_mmcif_dir,
                            FLAGS.max_template_date,
                            FLAGS.obsolete_pdbs_path,
                            FLAGS.pdb70_database_path,
                            FLAGS.uniref90_database_path,
                            FLAGS.mgnify_database_path,
                            FLAGS.bfd_database_path,
                            FLAGS.uniref30_database_path,
                            FLAGS.small_bfd_database_path,
                            FLAGS.use_precomputed_msas,
                            FLAGS.uniprot_database_path,
                            run_multimer_system,
                            fasta_path)

        print("Return status of feature dict creation ", fut.result())

        timings, unrelaxed_proteins, unrelaxed_pdbs, ranking_confidences, label = predict_structure(
            fasta_path=fasta_path,
            fasta_name=fasta_name,
            output_dir=output_dir,
            model_names=model_names,
            random_seed=random_seed)

        structure_ranker(output_dir=output_dir,
                         fasta_name=fasta_name,
                         models_to_relax=FLAGS.models_to_relax,
                         timings=timings,
                         unrelaxed_proteins=unrelaxed_proteins,
                         unrelaxed_pdbs=unrelaxed_pdbs,
                         ranking_confidences=ranking_confidences,
                         label=label,
                         use_amber=FLAGS.use_amber)

        end_time = time.time()
        additional_timings[f'{fasta_name}_prediction_time'] = end_time - t_0
        additional_timings[f'{fasta_name}_prediction_start_time'] = datetime.datetime.fromtimestamp(t_0).strftime(
            "%H:%M:%S")
        additional_timings[f'{fasta_name}_prediction_end_time'] = datetime.datetime.fromtimestamp(end_time).strftime(
            "%H:%M:%S")

        additional_timings.update(total_time)
        logging.info('\n\n Additional timings for %s: %s', fasta_name, additional_timings)

        additional_timings_output_path = os.path.join(output_dir, 'additional_timings.json')
        with open(additional_timings_output_path, 'w') as f:
            f.write(json.dumps(additional_timings, indent=4))

    t_pred_end = time.time()
    total_time['prediction_time'] = t_pred_end - t_pred_start
    total_time['prediction_start_time'] = datetime.datetime.fromtimestamp(t_pred_start).strftime("%H:%M:%S")
    total_time['prediction_end_time'] = datetime.datetime.fromtimestamp(t_pred_end).strftime("%H:%M:%S")

    total_time['Total Prediction Time'] = t_pred_end - t_setup_start
    total_time['Total_start_time'] = datetime.datetime.fromtimestamp(t_setup_start).strftime("%H:%M:%S")
    total_time['Total_end_time'] = datetime.datetime.fromtimestamp(t_pred_end).strftime("%H:%M:%S")

    # Save this dict as json file
    logging.info('\n\n Total timing info: %s', total_time)

    total_time_output_path = os.path.join(FLAGS.output_dir, 'total_time.json')
    with open(total_time_output_path, 'w') as f:
        f.write(json.dumps(total_time, indent=4))

    logging.info("\n\n Alphafold Prediction Ended")


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'fasta_paths',
        'output_dir'
    ])
    app.run(main)
