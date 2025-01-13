from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Union, Iterable
from typing_extensions import Self
from pathlib import Path
import pandas as pd

from openadmet_toolkit.cheminf.rdkit_funcs import run_reaction, smiles_to_inchikey

class ReactionSMART(BaseModel):
    reaction: str = Field(..., title="Reaction SMARTS", description="The reaction SMARTS string")
    reaction_name: str = Field(..., title="Name", description="The name of the reaction")
    product_names : list[str] = Field(..., title="Fragment names", description="The names of the products")
    reactant_names: list[str] = Field(..., title="Reactant names", description="The names of the reactants")

    def reactants(self): 
        reactant_side = self.reaction.split(">>")[0]
        # split into components on the "." token and return the list
        return reactant_side.split(".")

    def products(self):
        product_side = self.reaction.split(">>")[1]
        # split into components on the "." token and return the list
        return product_side.split(".")

    
    @model_validator(mode='after')
    def check_product_reactant_names(self) -> Self:
        if len(self.product_names) != len(self.products()):
            raise ValueError(f"Number of product names {len(self.product_names)} does not match number of products {len(self.products())}")
        if len(self.reactant_names) != len(self.reactants()):
            raise ValueError(f"Number of reactant names {len(self.reactant_names)} does not match number of reactants {len(self.reactants())}")
        return self

    
    


class BuildingBlockCatalouge(BaseModel):
    building_block_csv: Union[str,Path] = Field(..., title="CSV files", description="The CSV file containing the building block information")
    smiles_column: str = Field("SMILES", title="SMILES column", description="The name of the column containing the SMILES strings")
    inchikey_column: str = Field("INCHIKEY", title="INCHIKEY column", description="The name of the column containing the INCHIKEY strings")
    vendor_column: str = Field("VENDOR", title="Vendor column", description="The name of the column containing the vendor information")
    subselect_vendors: list[str] = Field(None, title="Vendor subselection", description="The list of vendors to subselect")
    
    @model_validator(mode='after')
    def check_columns(self) -> Self:
        df = pd.read_csv(self.building_block_csv)
        if self.smiles_column not in df.columns:
            raise ValueError(f"Column {self.smiles_column} not found in {self.building_block_csv}")
        if self.inchikey_column not in df.columns:
            raise ValueError(f"Column {self.inchikey_column} not found in {self.building_block_csv}")
        return self

    


    def load(self) -> pd.DataFrame:
        df =  pd.read_csv(self.building_block_csv)
        if self.subselect_vendors is not None:
            df = df[df[self.vendor_column].isin(self.subselect_vendors)]
        return df





class ForwardRetrosynth(BaseModel):
    reaction: ReactionSMART
    building_blocks: BuildingBlockCatalouge

    def run(self) -> list[str]:
        pass


class Retrosynth(BaseModel):
    reaction: ReactionSMART = Field(..., title="Reaction", description="The reaction to run")

    def run(self, df: pd.DataFrame, smiles_column:str="SMILES") -> list[str]:
        if smiles_column not in df.columns:
            raise ValueError(f"Column {self.smiles_column} not found in dataframe")
        df[f"{self.reaction.reaction_name}_products"] = df[smiles_column].apply(lambda x: run_reaction(x, self.reaction.reaction))
        # find the ones that are not NA and we will annotate them with the product names
        for i, product_name in enumerate(self.reaction.product_names):
            df[f"{self.reaction.reaction_name}_{product_name}"] = df[f"{self.reaction.reaction_name}_products"].apply(lambda x: x[i] if x is not pd.NA else pd.NA)
            # add inchikeys for the products also
            df[f"{self.reaction.reaction_name}_{product_name}_inchikey"] = df[f"{self.reaction.reaction_name}_{product_name}"].apply(lambda x: smiles_to_inchikey(x) if x is not pd.NA else pd.NA)
        
        return df
    

class BuildingBlockLibrarySearch(BaseModel):
    reaction: ReactionSMART
    building_blocks: BuildingBlockCatalouge

    def run(self, df: pd.DataFrame, smiles_column:str="SMILES") -> list[str]:
        # create retrosynthesis object
        retrosynth = Retrosynth(reaction=self.reaction)
        # run the retrosynthesis
        df = retrosynth.run(df, smiles_column)
        
        # drop the columns that are NA for the products
        df = df.dropna(subset=f"{self.reaction.reaction_name}_products")

        # load the building block library
        building_block_df = self.building_blocks.load()

        # now we will search for the building blocks in the library, building up a mask for each product if its INCHEKEY is in the library INCHIKEY column
        # we will then combine the masks to get a final mask for the library

        combined_mask = None

        for i, product_name in enumerate(self.reaction.product_names):
            # get the inchikeys for the products
            inchikeys = df[f"{self.reaction.reaction_name}_{product_name}_inchikey"]
            mask = inchikeys.isin(building_block_df[self.building_blocks.inchikey_column])
            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = combined_mask & mask
            
        df["vendor_synthesis"] = combined_mask

        # ok now drop the rows that cannot be synthesized
        df = df[df["vendor_synthesis"]]

        # now search the library


        return df





        # now we will search for the building blocks in the library




        









        



