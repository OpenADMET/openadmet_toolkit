from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel, Field, field_validator
from rdkit import Chem
from rdkit.rdBase import BlockLogs
from rdkit_funcs import smiles_to_inchikey, standardize_smiles
from tqdm import tqdm


class CSVProcessing(BaseModel):
    """
    Class to handle processing data from a csv downloaded
    """

    csv_path: Path = Field(..., description="Path to the ChEMBL csv")

    def __init__(self, **data):
        super().__init__(**data)

    @classmethod
    def read_csv(csv_path, sep=","):
        return pd.read_csv(csv_path, sep)

    @classmethod
    def standardize_smiles_and_convert(self):
        with BlockLogs():
            self.chemblData["CANONICAL_SMILES"] = self.chemblData[
                "Smiles"
            ].progress_apply(lambda x: standardize_smiles(x))
        with BlockLogs():
            self.chemblData["INCHIKEY"] = self.chemblData[
                "CANONICAL_SMILES"
            ].progress_apply(lambda x: smiles_to_inchikey(x))
        self.chemblData.dropna(subset="INCHIKEY", inplace=True)


class ChEMBLProcessing(CSVProcessing):
    """
    Class to handle processing data from a csv downloaded
    from ChEMBL

    """

    def __init__(self, **data):
        super().__init__(**data)
        self.chemblData = self.read_csv(self.csv_path, ";")
        self.keep_cols = [
            "CANONICAL_SMILES",
            "INCHIKEY",
            "pChEMBL mean",
            "pChEMBL std",
            "Molecule Name",
            "assay_count",
            "Action Type",
        ]
        self.standardize_smiles_and_convert()

    @classmethod
    def select_quality_data_inhibition(
        self, N: int = 10, pchembl_thresh: float = 5.0, L: int = 1, save_as=None
    ):
        better_assay = self.chemblData[
            (self.chemblData["Standard Type"] == "IC50")
            | (self.chemblData["Standard Type"] == "AC50")
            | (self.chemblData["Standard Type"] == "pIC50")
            | (self.chemblData["Standard Type"] == "XC50")
            | (self.chemblData["Standard Type"] == "EC50")
            | (self.chemblData["Standard Type"] == "Ki")
            | (self.chemblData["Standard Type"] == "Potency")
        ]
        better_units = better_assay[better_assay["Standard Units"] == "nM"]
        combined = self.get_num_compounds_per_assay(better_units)

        more_than_N_compounds = self.get_more_than_N_compounds(combined)
        assays = more_than_N_compounds["Assay ChEMBL ID"].nunique()
        num_assays_per_compound_df = self.get_num_assays_per_compound(
            more_than_N_compounds
        )
        combined_2 = more_than_N_compounds.join(
            num_assays_per_compound_df, on="INCHIKEY"
        )
        combined_2.sort_values("assay_count", ascending=False, inplace=True)
        combined_2["assay_count"] = combined_2["assay_count"].astype(int)

        compound_grouped_mean = combined_2.groupby("INCHIKEY")["pChEMBL Value"].mean()
        compound_grouped_mean.reset_index()

        cgm = compound_grouped_mean.reset_index(name="pChEMBL mean")
        cgm = cgm.set_index("INCHIKEY")
        combined_3 = combined_2.join(cgm, on="INCHIKEY")

        compound_grouped_std = combined_2.groupby("INCHIKEY")["pChEMBL Value"].std()

        cgstd = compound_grouped_std.reset_index(name="pChEMBL std")
        cgstd = cgstd.set_index("INCHIKEY")
        combined_4 = combined_3.join(cgstd, on="INCHIKEY")

        # get active compounds
        # defined as compounds above pChEMBL value specified (default 5.0)
        active = combined_4[combined_4["pChEMBL mean"] >= pchembl_thresh]
        clean_deduped = self.clean_and_dedupe_actives(active, save_as, inhibition=True)
        return self.more_than_L_assays(clean_deduped, L)

    @classmethod
    def select_quality_data_reactivity(self, save_as):
        substrates = self.chemblData[self.chemblData["Action Type"] == "SUBSTRATE"]
        return self.clean_and_dedupe_activities(substrates, save_as, inhibition=False)

    @classmethod
    def more_than_L_assays(self, clean_deduped, L=1, save_as=None):
        more_than_eq_L_assay = clean_deduped[
            clean_deduped["appears_in_N_ChEMBL_assays"] >= L
        ]
        if save_as is not None:
            more_than_eq_L_assay.to_csv(save_as, index=False)
        return more_than_eq_L_assay.INCHIKEY.nunique()

    @classmethod
    def clean_and_dedupe_actives(self, active, save_as=None, inhibition=True):
        clean_active = active[self.keep_cols]
        clean_active.rename(
            columns={
                "assay_count": "appears_in_N_ChEMBL_assays",
                "Molecule Name": "common_name",
                "Action Type": "action_type",
            },
            inplace=True,
        )
        clean_active_sorted = clean_active.sort_values(
            ["common_name", "action_type"], ascending=[False, False]
        )  # keep the ones with names if possible
        clean_deduped = clean_active_sorted.drop_duplicates(
            subset="INCHIKEY", keep="first"
        )
        if inhibition:
            clean_deduped = clean_deduped.sort_values(
                "appears_in_N_ChEMBL_assays", ascending=False
            )
            clean_deduped["action_type"] = clean_deduped["action_type"].apply(
                lambda x: x.lower() if isinstance(x, str) else x
            )
        else:
            clean_deduped["action_type"] = "substrate"
        clean_deduped["dataset"] = "ChEMBL_curated"
        clean_deduped["active"] = True
        if save_as is not None:
            clean_deduped.to_csv(save_as, index=False)
        return clean_deduped

    @classmethod
    def get_more_than_N_compounds(self, combined):
        more_than_N_compounds = combined[combined["molecule_count"] > N]
        more_than_N_compounds.INCHIKEY = more_than_N_compounds.INCHIKEY.astype(str)
        return more_than_N_compounds["Assay ChEMBL ID"].nunique()

    @classmethod
    def get_num_assays_per_compound(self, more_than_N_compounds):
        num_assays_per_compound_df = (
            more_than_N_compounds.groupby(["INCHIKEY"])["Assay ChEMBL ID"]
            .size()
            .reset_index(name="assay_count")
        )
        return num_assays_per_compound_df.set_index("INCHIKEY")

    @classmethod
    def get_num_compounds_per_assay(self, better_units):
        num_compounds_per_assay = better_units.groupby("Assay ChEMBL ID")[
            "Molecule ChEMBL ID"
        ].nunique()
        num_compounds_per_assay_df = pd.DataFrame(num_compounds_per_assay)
        num_compounds_per_assay_df.rename(
            columns={"Molecule ChEMBL ID": "molecule_count"}, inplace=True
        )
        return better_units.join(num_compounds_per_assay_df, on="Assay ChEMBL ID")

    @classmethod
    def aggregate_activity(self, combined_2):
        compound_grouped_mean = combined_2.groupby("INCHIKEY")["pChEMBL Value"].mean()
        return compound_grouped_mean.reset_index()

    @classmethod
    def get_num_assays_per_compound(self, more_than_N_compounds):
        num_assays_per_compound_df = (
            more_than_N_compounds.groupby(["INCHIKEY"])["Assay ChEMBL ID"]
            .size()
            .reset_index(name="assay_count")
        )
        return num_assays_per_compound_df.set_index("INCHIKEY")


class PubChemProcessing(CSVProcessing):
    """
    Class to handle processing data from a csv downloaded
    from PubChem

    """

    def __init__(self, **data):
        super().__init__(**data)
        self.pubChemData = self.read_csv(self.csv_path)
        self.keep_cols = [
            "CANONICAL_SMILES",
            "INCHIKEY",
            "PUBCHEM_ACTIVITY_OUTCOME",
            "PUBCHEM_CID",
        ]
        self.delete_metadata_rows()
        self.pubChemData = self.pubChemData.dropna(subset="PUBCHEM_CID")
        self.pubChemData["PUBCHEM_SID"] = self.pubChemData["PUBCHEM_SID"].astype(int)
        self.pubChemData["PUBCHEM_CID"] = self.pubChemData["PUBCHEM_CID"].astype(int)
        self.standardize_smiles_and_convert()
        self.pubChemData.dropna(subset="INCHIKEY")

    @classmethod
    def delete_metadata_rows(self):
        to_del = 0
        for index, row in self.pubChemData.iterrows():
            if index == 0:
                continue
            elif Chem.MolFromSmiles(row["PUBCHEM_EXT_DATASOURCE_SMILES"]) is not None:
                to_del += 1
            else:
                break
        self.pubChemData = self.pubChemData.drop(
            labels=list(range(0, to_del)), axis=0
        ).reset_index(drop=True)

    @classmethod
    def clean_data_inhibition(self, aid, data_type, save_as=None):
        clean = self.pubChemData[self.keep_cols]
        clean["dataset"] = aid
        clean["data_type"] = data_type
        clean["active"] = clean["PUBCHEM_ACTIVITY_OUTCOME"] == "Active"
        clean["common_name"] = pd.NA
        clean["action_type"] = "inhibitor"
        if save_as is not None:
            clean.to_csv(save_as, index=False)
