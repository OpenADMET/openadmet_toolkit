from openadmet_toolkit.cheminf.data_curation import ChEMBLProcessing

path = "../../../../ChEMBL_CYP1A2_activities.csv"

chembl = ChEMBLProcessing(path)
print(chembl.select_quality_data_inhibition(save=False))
print(chembl.select_quality_data_reactivity(save=False))
