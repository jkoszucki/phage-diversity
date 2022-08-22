# import modules
import warnings
import sys
import shutil
import pandas as pd
from pathlib import Path
import markov_clustering as mc
import networkx as nx
import itertools
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from subprocess import run
import pyvips
import yaml
from PIL import Image, ImageOps, ImageFont, ImageDraw

from Bio import SeqIO
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import Phylo


def preprocessing(inphared_dir, prophages_dir, phrogs_annot_table, phagedb_dir):
    """ ... """

    # input paths
    prophages_fasta =  Path(prophages_dir, 'prophages.fasta')
    prophages_metadata = Path(prophages_dir, 'prophages.tsv')
    prophages_annot = Path(prophages_dir, 'functional-annotation/output/5_proteins.tsv')
    prophages_orfs_dir =  Path(prophages_dir, 'functional-annotation/1_orf_prediction/3_orfs')
    prophages_annot_input = Path(prophages_dir, 'functional-annotation/output/1_input.txt')


    inphared_fasta =  Path(inphared_dir, 'klebsiella_inphared.fasta')
    inphared_metadata = Path(inphared_dir, 'klebsiella_inphared.tsv')
    inphared_annot = Path(inphared_dir, 'functional-annotation/output/5_proteins.tsv')
    inphared_orfs_dir =  Path(inphared_dir, 'functional-annotation/1_orf_prediction/3_orfs')
    inphared_annot_input = Path(inphared_dir, 'functional-annotation/output/1_input.txt')

    # output paths
    phages_fasta =  Path(phagedb_dir, 'phages.fasta')
    phages_genank =  Path(phagedb_dir, 'phages.gb')
    phages_metadata = Path(phagedb_dir, 'phages.tsv')
    phages_annot_input = Path(phagedb_dir, 'annot_input.txt')
    phages_annot = Path(phagedb_dir, 'annot.tsv')

    split_records_dir = Path(phagedb_dir, 'split_records')
    phages_orfs_dir = Path(split_records_dir, 'orfs')
    phages_genbank_dir = Path(split_records_dir, 'genbank')
    phages_fasta_dir = Path(split_records_dir, 'fasta')

    if split_records_dir.exists(): shutil.rmtree(split_records_dir) # remove if exist directory
    Path(phages_orfs_dir).mkdir(exist_ok=True, parents=True) # create empty directory
    Path(phages_genbank_dir).mkdir(exist_ok=True, parents=True) # create empty directory
    Path(phages_fasta_dir).mkdir(exist_ok=True, parents=True) # create empty directory

    # copy to one files all phages
    records = [SeqIO.parse(path, 'fasta') for path in [prophages_fasta, inphared_fasta]]
    records = list(itertools.chain(*records))
    n = SeqIO.write(records, phages_fasta, 'fasta')

    for r in records:
        path = Path(phages_fasta_dir, r.name + '.fasta')
        SeqIO.write(r, path, 'fasta')

    if n == len(records): print('Records saved successfully :) ')
    else: print('Some records are missing.')

    # copy orf files
    orf_files = [list(dir.glob('*fasta')) for dir in [prophages_orfs_dir, inphared_orfs_dir]]
    orf_files = list(itertools.chain(*orf_files))

    for path in orf_files:
        shutil.copy(path, phages_orfs_dir)
    print('ORFs copied successfully :)')

    # get one annotation table
    prophages_annot_df = pd.read_csv(prophages_annot, sep='\t')
    inphared_annot_df = pd.read_csv(inphared_annot, sep='\t')
    annot_df = pd.concat([prophages_annot_df, inphared_annot_df])
    annot_df.to_csv(phages_annot, sep='\t', index=False)

    # from annotation table to genbank files
    table2genbank(phages_annot, phages_fasta, phages_genank, phrogs_annot_table, colour_type='structural')

    # load genbank file and save each record to seperate file for easyfig
    records = SeqIO.parse(phages_genank, 'genbank')
    for r in records:
        path = Path(phages_genbank_dir, r.name + '.gb')
        SeqIO.write(r, path, 'genbank')

    # combining metadata tables
    inphared_df = pd.read_csv(inphared_metadata, sep='\t')
    prophages_df = pd.read_csv(prophages_metadata, sep='\t')

    prophages_df['prophage'] = True
    inphared_df['prophage'] = False

    inphared_df['Genome Length (bp)'] = np.round(inphared_df['Genome Length (bp)'] / 1000, 2)

    prophages_df.rename({'phage_length': 'Genome Length (kb)'}, axis=1, inplace=True)
    inphared_df.rename({'Accession': 'phageID'}, axis=1, inplace=True)
    metadata_df = pd.concat([prophages_df, inphared_df])
    metadata_df.drop('n', axis=1, inplace=True)

    metadata_df.reset_index(drop=True)
    metadata_df.index = metadata_df.index + 1
    metadata_df['n'] = metadata_df.index

    cols = ['n']+ list(metadata_df.columns[:-1])
    metadata_df[cols].to_csv(phages_metadata, sep='\t', index=False)
    print('Metadata unified and copied successfully :) ')

    annot_input_dfs = [pd.read_csv(path, sep='\t') for path in [prophages_annot_input, inphared_annot_input]]
    annot_input_df = pd.concat(annot_input_dfs)
    annot_input_df.to_csv(phages_annot_input, sep='\t', index=False)
    print('Functiona annotation input tables merged and saved :)')


def run_ANImm(animm_dir, phagedb_dir, output_dir):
    """ ... """

    if Path(output_dir).exists():
        print(f'ANImm already done! To rerun delete folder: {output_dir}')
        return
    else: pass

    config_path = Path(animm_dir, 'cds-based.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['input_dir'] = str(Path(phagedb_dir, 'split_records', 'orfs'))
    config['output_dir'] = str(output_dir)

    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    cmd = f'cd {animm_dir}; snakemake --use-conda --cores all --configfile cds-based.yml'
    process = run(cmd, capture_output=True, shell=True)

    return process


def get_phariants(wgrr_file, input_table, output_file, wgrr_threshold):
    """ ... """

    clusters_df = wgrr_clusters(wgrr_file, output_file, wgrr_threshold)
    clusters_df.rename({'member': 'contigID'}, axis=1, inplace=True)

    input_df = pd.read_csv(input_table, sep='\t')
    clusters_df = clusters_df.merge(input_df, on='contigID', how='left')

    clusters_df.loc[clusters_df.groupby('phariantID')['contig_len [bp]'].idxmax(), 'status'] = 'repr'
    clusters_df['status'].fillna('member', inplace=True)
    clusters_df.to_csv(output_file, sep='\t', index=False)

    nrepr = clusters_df.loc[clusters_df['status'] == 'repr'].shape[0]
    print(f'Done! With wGRR treshold = {wgrr_threshold} we have {nrepr} phage clusters :)', end='')
    return clusters_df


def run_mashtree(phagedb_dir, output_dir):
    """ .... """

    fasta_dir = Path(phagedb_dir, 'split_records/fasta')
    output_file = str(Path(output_dir, 'tree.newick'))

    Path(output_dir).mkdir(exist_ok=True, parents=True)

    cmd = f'source ~/.bashrc; conda activate mashtree; mashtree.pl {fasta_dir}/* >> {output_file}; conda activate mybase;'

    print('Just run the command in bash. Problem with conda env AGAIN : / \n')
    print(cmd)

    # process = run(cmd, capture_output=True, shell=True)
    # if not str(process.stderr): print('Mashtree done! :)')

    return cmd


def tree2clades(mash_tree_path, phariants_table, clades, n_clusters=1, kmeans_show=True):
    """ ... """

    clusters_df = pd.read_csv(phariants_table, sep='\t')
    representatives = clusters_df.loc[clusters_df['status'] == 'repr', 'contigID'].to_list()

    tree = list(Phylo.parse(mash_tree_path, "newick"))[0]
    leafs = [leaf.name.upper() for leaf in tree.get_terminals() if leaf.name.upper() in representatives]
    total_branch_lengths = [(tree.distance(leaf), i) for i, leaf in enumerate(tree.get_terminals()) if leaf.name.upper() in representatives]

    if len(leafs) == len(representatives) and len(total_branch_lengths) == len(representatives):
        print(f'Everything is good :) Cluster tree with kmeans method (n_cluster: {n_clusters}).',)
    else: print('Something is wrong! Aborting! Check tree2clades function in utils.py!')

    x, y = list(zip(*total_branch_lengths))
    df = pd.DataFrame({'total_branch_length': x, 'n': y})
    kmeans = KMeans(n_clusters=n_clusters).fit(df)

    phage_clusters_df = pd.DataFrame({'clusterID': kmeans.labels_, 'contigID': leafs})
    phage_clusters_df['clusterID'] = phage_clusters_df['clusterID'] + 1
    phage_clusters_df.to_csv(clades, index=False, sep='\t')

    print('Done! Phages grouped using mashtree! :)', end='')

    if kmeans_show:
        plt.figure(figsize=(30,30))
        plt.scatter(df['total_branch_length'], df['n'], c= kmeans.labels_.astype(float), s=100, alpha=0.5)
        plt.show()
    else: pass

    return phage_clusters_df


def table2genbank(annot_table, phages_fasta, genbank, phrogs_annot_table, colour_type='structural'):
    """ Converts output table (nr 5) from functional annotation to gebank file.
    phages_fasta is a path to file with all phage sequences.
    Decide which colour to put in genbank: 'structural' or 'phrogs'."""

    if genbank.exists():
        print(f'Nothing to do in table2genbank :)\nTo force rerun delete {genbank}')
        return None
    else: pass

    annot_df = merge_annot_table_and_phrogs_metatable(annot_table, phrogs_annot_table)
    phages_records_fasta = list(SeqIO.parse(phages_fasta, 'fasta'))

    contigIDs = annot_df['contigID'].unique()

    records = []

    for contigID in contigIDs:

        for phage_record in phages_records_fasta:
            if phage_record.id == contigID:
                record = phage_record
                break

        # checkpoint
        if not record: print('phageID (contigID) from annotation table was not found in fasta records!')

        # if flipped: record.seq = record.seq.reverse_complement()

        # take subset of table with features only for one phage
        features = []
        filt_phage = (annot_df['contigID'] == contigID)
        phage_df = annot_df.loc[filt_phage].copy()

        protID_col, seqs_col, status_col = phage_df['proteinID'], phage_df['protein'], phage_df['status']
        start_col, stop_col, strand_col = phage_df['start'], phage_df['stop'], phage_df['strand']
        phrog_col, annot_col, cat_col, struct_col = phage_df['phrog'], phage_df['annot'], phage_df['category'], phage_df['structural']

        if colour_type == 'structural': color_col = phage_df['rgb_color_structural']
        elif colour_type == 'phrogs': color_col = phage_df['rgb_color_phrogs']
        else:
            print(f" ### Wrong colour type: {colour_type}!  Choose: 'structural' or 'phrogs'! ### ")
            exit()

        prots = list(zip(protID_col, seqs_col, status_col, start_col, stop_col, strand_col, phrog_col, annot_col, cat_col, struct_col, color_col))

        # iterate over proteins and create features
        for protID, seq, status, start, stop, strand, phrog, annot, cat, struct, color in prots:

            f = SeqFeature(FeatureLocation(start-1, stop, strand=int(f'{strand}1')), type='CDS')
            qualifiers = {
                    'db_xref': [f'phrog_{phrog}'],
                    'note': [cat],
                    'product': [annot],
                    'structural': [struct],
                    'trans_table': [11],
                    'translation': [seq],
                    'colour': color,
                    }

            f.qualifiers = qualifiers
            features.append(f)

        # add record data
        record.annotations['molecule_type'] = 'DNA'
        record.features = features
        record.id = contigID
        record.name = ''
        record.description = ''
        record.organism = 'phage'

        records.append(record)

    # Write the phage with appended protein to the genbank file.
    n = SeqIO.write(records, genbank, 'genbank')

    print(f'Done! {n} phages converted to genbank file.')

    return annot_df


def wgrr_clusters(wgrr_input, clusters_output, wgrr_threshold=0.9):

    wgrr_df = pd.read_csv(wgrr_input)

    print('Check! In some case I can loose singletons here!')
    phageIDs = list(set(wgrr_df['Seq1'].to_list() + wgrr_df['Seq2'].to_list()))
    filt_wgrr = (wgrr_df['wgrr'] >= wgrr_threshold)
    wgrr_df = wgrr_df.loc[filt_wgrr].loc[:, ['Seq1', 'Seq2']].copy()

    phage_nodes = list(set(wgrr_df['Seq1'].to_list() + wgrr_df['Seq2'].to_list()))

    # get missing singletons
    singletons = []
    for phageID in phageIDs:
        if phageID not in phage_nodes:
            singletons.append([phageID])

    # use networkx to generate the graph
    G = nx.Graph()
    G.add_edges_from(wgrr_df.to_numpy())

    # then get the adjacency matrix (in sparse form)
    matrix = nx.to_scipy_sparse_array(G)
    result = mc.run_mcl(matrix, inflation = 2)           # run MCL with default parameters
    clusters = mc.get_clusters(result)    # get clusters

    # map nodes idx to names and sort
    clusters.sort(key=lambda cluster: len(cluster), reverse=True)
    nodes = list(G.nodes)

    mapped_clusters = []
    for nodes_indicies in clusters:
        cluster_nodes = list(map(nodes.__getitem__, nodes_indicies))
        mapped_clusters.append(cluster_nodes)

    # add singletons
    mapped_clusters= mapped_clusters + singletons

    clusterIDs = []
    for i, cluster in enumerate(mapped_clusters):
        clusterIDs.append(len(cluster) * [i + 1])

    mapped_clusters = list(itertools.chain(*mapped_clusters))
    clusterIDs = list(itertools.chain(*clusterIDs))

    clusters_df = pd.DataFrame({'phariantID': clusterIDs, 'member': mapped_clusters})
    clusters_df.to_csv(clusters_output, sep='\t', index=False)

    return clusters_df


def merge_annot_table_and_phrogs_metatable(annot_table, phrogs_annot_table):
    """ merge output table from functional annotation (nr 5) for each protein with phrogs metatable """

    annot_df = pd.read_csv(annot_table, sep='\t')
    phrogs_df = pd.read_csv(phrogs_annot_table, sep='\t')

    annot_df['phrog'] = annot_df.apply(lambda row: int(row['target'].strip('phrog_')), axis=1)
    annot_df = annot_df.merge(phrogs_df, on='phrog', how='left')

    return annot_df


def run_easyfig(work_dir, clades, prophages, phages_genbank_dir, easyfig_script, leg_name='structural', annotate_columns=['K_locus', 'ST', 'phageID', 'genetic_localisation', 'ICTV_Species', 'ICTV_Genus'], font_path='other/arial.ttf'):
    """ ... """

    clades_df = pd.read_csv(clades, sep='\t')

    easyfig_dir = Path(work_dir, 'easyfig')
    raw_dir = Path(easyfig_dir, 'raw')
    annotated_dir = Path(easyfig_dir, 'annotated')
    print(str(Path(font_path).resolve()))
    font = ImageFont.truetype(str(Path(font_path).resolve()), 70)
    clusters = sorted(list(clades_df['clusterID'].unique()))

    # checkpoint
    if annotated_dir.exists():
        print('Nothing to do :)')
        print(f'To force rerun delete {str(annotated_dir)}')
        return None, None
    else:
        print('Generating easyfig figures for {len(clusters)} clusters :) ', sep='')

    raw_dir.mkdir(exist_ok=True, parents=True)
    annotated_dir.mkdir(exist_ok=True, parents=True)

    print('Subset!!!')
    for nclust in clusters[0:2]:
        leafs = clades_df.loc[clades_df['clusterID'] == nclust]['contigID'].to_list()
        prophages_df = pd.read_csv(prophages, sep='\t')
        prophages_df.fillna('', inplace=True)

        # convert phage/leaf names to paths to genbank files
        easyfig_input_files = [str(Path(phages_genbank_dir, leaf.upper() + '.gb')) for leaf in leafs]
        easyfig_input_files = ' '.join(easyfig_input_files)

        svg = Path(raw_dir, f'clade_{str(nclust)}.svg')

        cmd = f'source ~/.bashrc; conda activate easyfig ; \
                python2 {easyfig_script} \
                -svg -legend double -leg_name {leg_name} -f CDS -f1 T -i 60 -filter -aln right \
                -o {str(svg)} {easyfig_input_files};'
        print(cmd)
        # complete_process = run(cmd, capture_output=True, shell=True)
        #
        # # convert svg2jpeg
        # jpeg_raw = Path(raw_dir, svg.stem + '.jpg')
        # image = pyvips.Image.new_from_file(svg, dpi=100).flatten(background=255)
        # image.write_to_file(jpeg_raw)
        #
        # jpeg_annot = Path(annotated_dir, svg.stem + '.jpg')
        #
        # lines_to_annotate = get_text(prophages_df, leafs, annotate_columns)
        #
        # color = (0,0,0)
        # add_annotation(jpeg_raw, jpeg_annot, font, color, lines_to_annotate)

    print('Done! :)', sep='')
    return annotated_dir, complete_process


def get_text(df, phageIDs, columns):
    """ ... """

    df['phageID'] = df.apply(lambda row: ''.join([str(row['n']) + '_' + row['phageID']]), axis=1)
    df = df.loc[df['phageID'].isin(phageIDs)].copy()

    df['phageID_cat'] = pd.Categorical(df['phageID'], categories=phageIDs, ordered=True)
    df.sort_values('phageID_cat', inplace=True)

    lines_to_annotate = ['  '.join(line) for line in df[columns].values.tolist()]

    return lines_to_annotate


def add_border(input_image, border, color):
    """ ... """
    img = Image.open(input_image)

    if isinstance(border, int) or isinstance(border, tuple):
        bimg = ImageOps.expand(img, border=border, fill=color)
    else:
        raise RuntimeError('Border is not an integer or tuple!')

    return bimg


def add_annotation(input_image, output_image, font, color, text_lines):
    """ Color as rgb tuple."""

    add_pixels = 2500
    img = add_border(input_image, (0, 0, add_pixels, 0), color='white')
    draw = ImageDraw.Draw(img)
    xlocation, ylocation = img.size[0] - add_pixels + 100, - 10

    for text in text_lines:
        loc = (xlocation, ylocation)
        draw.text(loc, text, color, font)
        ylocation = ylocation + 208

    img.save(output_image)
