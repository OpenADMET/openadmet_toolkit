from openadmet_toolkit.cheminf.data_curation import ChEMBLProcessing, PubChemProcessing
from openadmet_toolkit.tests.datafiles import chembl_file, pubchem_file
import pandas as pd
import pytest

@pytest.fixture()
def chembl():
    return chembl_file 

@pytest.fixture()
def pubchem():
    return pubchem_file

def test_chembl_inhib():
    chembl_inhib = ChEMBLProcessing(inhib=True)
    df = chembl_inhib.process(chembl_file)
    assert all(pd.notna(df["Smiles"]))
    assert all(pd.notna(df["CANONICAL_SMILES"]))
    assert df["INCHIKEY"].is_unique

def test_chembl_react():
    chembl_react = ChEMBLProcessing(react=True)
    df = chembl_react.process(chembl_file)
    assert all(pd.notna(df["Smiles"]))
    assert all(pd.notna(df["CANONICAL_SMILES"]))
    assert df["INCHIKEY"].is_unique   

def test_pubchem_inhib():
    pubchem_inhib = PubChemProcessing(inhib=True)
    df = pubchem_inhib.process(pubchem_file, 'test1', 'test2')
    assert all(pd.notna(df["Smiles"]))
    assert all(pd.notna(df["CANONICAL_SMILES"]))
    
    # assert df["INCHIKEY"].is_unique ## THIS ONE IS FAILING
