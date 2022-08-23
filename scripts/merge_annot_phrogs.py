import pandas as pd


def merge_annot_table_and_phrogs_metatable(annot_table, phrogs_annot_table):
    """ merge output table from functional annotation (nr 5) for each protein with phrogs metatable """

    annot_df = pd.read_csv(annot_table, sep='\t')
    phrogs_df = pd.read_csv(phrogs_annot_table, sep='\t')

    annot_df['phrog'] = annot_df.apply(lambda row: int(row['target'].strip('phrog_')), axis=1)
    annot_df = annot_df.merge(phrogs_df, on='phrog', how='left')

    return annot_df


annot_table = '/home/MCB/jkoszucki/phagedb/PHAGES-DB/0_phagedb/annot.tsv'
phrogs_annot_table = '/home/MCB/jkoszucki/Code/phage-diversity/other/upgraded_phrog_annot_v3.tsv'

annot_phrogs_df = merge_annot_table_and_phrogs_metatable(annot_table, phrogs_annot_table)
annot_phrogs_df.to_csv('annot_phrogs.tsv', sep='\t', index=False)

