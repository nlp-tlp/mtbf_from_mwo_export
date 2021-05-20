'''
  Simple file for running pytest tests.

  To run tests from a command line:
  > /project_dir> python -m pytest --cov=pipeline
'''
from pytest import raises
from pipeline import term_expansion, collocations, mwo_to_mtbf

import os


def test_term_expansion(mocker):
  '''Test term_expansion module on its own.
  '''

  config_path = os.path.join('test_data', 'test_config.yml')

  assert os.path.exists(config_path)

  # Setup mocks to simulate user interaction
  mocker.patch('pipeline.term_expansion.get_replace_terms',
               return_value='replace')
  mocker.patch('pipeline.term_expansion.get_repair_terms',
               return_value='repair')
  mocker.patch('pipeline.term_expansion.get_new_terms', return_value='')

  term_expansion.controller(
      config_path)    # Uses w2v and SME to output token_file.xlsx


def test_collocations(mocker):
  '''Test collocations module on its own
  '''

  config_path = os.path.join('test_data', 'test_config.yml')

  assert os.path.exists(config_path)

  # Setup mocks to simulate user interaction
  mocker.patch('pipeline.collocations.get_decision_terms', return_value='n')

  collocations.controller(config_path)    # Adds collocations to token file


def test_mwo_to_mtbf():
  '''Test mwo_to_mtbf module on its own

  Notes: This is expected to fail with a ValueError
  '''

  config_path = os.path.join('test_data', 'test_config.yml')

  assert os.path.exists(config_path)

  with raises(ValueError) as excinfo:

    mwo_to_mtbf.controller(
        config_path)    # Performs parameter estimation over MWOs

  assert "'FUNCTIONAL_LOC' is both an index level and a column label, which is ambiguous." in str(
      excinfo.value)
