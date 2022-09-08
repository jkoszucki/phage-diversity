# import modules
import warnings
import sys
import ast
import shutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

    if Path(phagedb_dir).exists():
        print(f'Preprocessing alread done! To rerun delete folder: {phagedb_dir}')
        return None
    else: pass

    rbps_table = '/home/MCB/jkoszucki/phagedb/byhand/domains_detections.csv'
    backphlip = '/home/MCB/jkoszucki/phagedb/byhand/backphlip/phages.fasta.bacphlip'
    backphlip_df = pd.read_csv(backphlip, sep='\t')

    print(f'Loading backphlip from {backphlip}... ', end='')
    backphlip_df.loc[backphlip_df['Virulent'] >= 0.75, 'backphlip'] = 'virulent'
    backphlip_df.loc[backphlip_df['Virulent'] <= 0.25, 'backphlip'] = 'temperate'

    filt_strong = (backphlip_df['Virulent'] >= 0.95) | (backphlip_df['Virulent'] <= 0.05)
    filt_weak = ~(filt_strong) & (backphlip_df['Virulent'] >= 0.75) | (backphlip_df['Virulent'] <= 0.25)

    backphlip_df.loc[filt_strong, 'backphlip_conf'] = 'strong'
    backphlip_df.loc[filt_weak, 'backphlip_conf'] = 'weak'
    backphlip_df[['backphlip', 'backphlip_conf']] = backphlip_df[['backphlip', 'backphlip_conf']].fillna('none')

    backphlip_df.columns = ['phageID', 'virulent', 'temperate', 'backphlip', 'backphlip_conf']
    backphlip_df = backphlip_df[['phageID', 'backphlip', 'backphlip_conf']]
    print('Done!')

    # input paths
    prophages_fasta =  Path(prophages_dir, 'prophages.fasta')
    prophages_metadata = Path(prophages_dir, 'prophages.tsv')
    prophages_annot = Path(prophages_dir, 'functional-annotation/output/5_proteins.tsv')
    prophages_proteins_dir =  Path(prophages_dir, 'functional-annotation/1_orf_prediction/4_proteins')
    prophages_annot_input = Path(prophages_dir, 'functional-annotation/output/1_input.txt')


    inphared_fasta =  Path(inphared_dir, 'klebsiella_inphared.fasta')
    inphared_metadata = Path(inphared_dir, 'klebsiella_inphared.tsv')
    inphared_annot = Path(inphared_dir, 'functional-annotation/output/5_proteins.tsv')
    inphared_proteins_dir =  Path(inphared_dir, 'functional-annotation/1_orf_prediction/4_proteins')
    inphared_annot_input = Path(inphared_dir, 'functional-annotation/output/1_input.txt')

    # output paths
    phages_fasta =  Path(phagedb_dir, 'phages.fasta')
    phages_genbank =  Path(phagedb_dir, 'phages.gb')
    phages_metadata = Path(phagedb_dir, 'phages.tsv')
    phages_annot_input = Path(phagedb_dir, 'annot_input.txt')
    phages_annot = Path(phagedb_dir, 'annot.tsv')
    phages_annot_phrogs = Path(phagedb_dir, 'annot_phrogs.tsv')

    split_records_dir = Path(phagedb_dir, 'split_records')
    phages_proteins_dir = Path(split_records_dir, 'proteins')
    phages_genbank_dir = Path(split_records_dir, 'genbank')
    phages_fasta_dir = Path(split_records_dir, 'fasta')

    if split_records_dir.exists(): shutil.rmtree(split_records_dir) # remove if exist directory
    Path(phages_proteins_dir).mkdir(exist_ok=True, parents=True) # create empty directory
    Path(phages_genbank_dir).mkdir(exist_ok=True, parents=True) # create empty directory
    Path(phages_fasta_dir).mkdir(exist_ok=True, parents=True) # create empty directory

    # copy to one files all phages
    records = [SeqIO.parse(path, 'fasta') for path in [prophages_fasta, inphared_fasta]]
    records = list(itertools.chain(*records))
    n = SeqIO.write(records, phages_fasta, 'fasta')

    # if n == len(records): print('Records saved successfully :) ')
    # else: print('Some records are missing.')

    # copy orf files
    proteins_files = [list(dir.glob('*fasta')) for dir in [prophages_proteins_dir, inphared_proteins_dir]]
    proteins_files = list(itertools.chain(*proteins_files))

    for path in proteins_files:
        shutil.copy(path, phages_proteins_dir)
    print('Proteins files per phage copied successfully (ANImm input) :)')

    # get one annotation table
    prophages_annot_df = pd.read_csv(prophages_annot, sep='\t')
    inphared_annot_df = pd.read_csv(inphared_annot, sep='\t')
    annot_df = pd.concat([prophages_annot_df, inphared_annot_df])
    annot_df.to_csv(phages_annot, sep='\t', index=False)

    # from annotation table to genbank files
    annot_phrogs_df = table2genbank(phages_annot, phages_fasta, phages_genbank, phrogs_annot_table, rbps_table, colour_type='structural')
    annot_phrogs_df.to_csv(phages_annot_phrogs, sep='\t', index=False)

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

    annot_input_dfs = [pd.read_csv(path, sep='\t') for path in [prophages_annot_input, inphared_annot_input]]
    annot_input_df = pd.concat(annot_input_dfs)
    annot_input_df.to_csv(phages_annot_input, sep='\t', index=False)
    print('Functional annotation input tables merged and saved :)')

    # add columns if phage should be flipped
    metadata_df = flip_phage(annot_phrogs_df, annot_input_df, metadata_df)
    metadata_df = metadata_df.merge(backphlip_df, on='phageID', how='left')
    phages2flip = metadata_df.loc[metadata_df['flip_phage'], 'phageID'].to_list()

    # load genbank file and save each record to seperate file for easyfig
    records = SeqIO.parse(phages_genbank, 'genbank')
    for r in records:
        if r.name in phages2flip: r = flip_record(r)
        else: pass

        # save genbank
        path = Path(phages_genbank_dir, r.name + '.gb')
        SeqIO.write(r, path, 'genbank')

        # save fasta
        path = Path(phages_fasta_dir, r.name + '.fasta')
        SeqIO.write(r, path, 'fasta')

    print('Phage flipped and saved to seperate genbank & fasta files :) ')

    # make table nice
    metadata_df.reset_index(drop=True)
    metadata_df.index = metadata_df.index + 1
    metadata_df['n'] = metadata_df.index
    cols = ['n'] + list(metadata_df.columns[:-1])

    metadata_df[cols].to_csv(phages_metadata, sep='\t', index=False)
    print('Metadata unified and copied successfully :) ')


def flip_phage(annot_phrogs_df, annot_input_df, metadata_df):
    """ add column to metadata df if phage should be flipped or not """

    # phage flip table beased on overall genes directionality
    mapper = {'+': 1, '-': -1}
    annot_phrogs_df['tmp_strand'] = annot_phrogs_df['strand'].map(mapper)
    directionality_df = annot_phrogs_df.groupby('contigID')['tmp_strand'].sum().reset_index()
    directionality_df.columns = ['contigID', 'directionality']

    directionality_df.loc[directionality_df['directionality'] < 0, 'flip_phage'] = True
    directionality_df['flip_phage'].fillna(False, inplace=True)

    directionality_df.columns = ['phageID', 'total_strand', 'flip_phage']

    metadata_df = metadata_df.merge(directionality_df, on='phageID', how='left')
    return metadata_df


def flip_record(record):
    """ Flip record. """

    length = len(record.seq)
    features = record.features

    # flip location of features
    new_features = []
    for f in features[::-1]:
        if f.strand == 1: strand = -1
        else: f.strand = strand = 1

        new_start = length - int(f.location.end)
        new_end = length - int(f.location.start)

        new_location = FeatureLocation(new_start, new_end, strand)
        f.location = new_location

        new_features.append(f)

    # flip sequence
    record.seq = record.seq.reverse_complement()
    record.features = new_features

    return record


def addRBPs(rbps_table, annot_df, rgb_color='209 29 83', html_color='#d11d53'):
    """ add the rbps to annot table """

    rbps_df = pd.read_csv(rbps_table)

    rbps_df['domains'] = rbps_df.apply(lambda row: ' '.join(ast.literal_eval(row['N_blocks']) + ast.literal_eval(row['C_blocks'])), axis=1)
    rbps_df = rbps_df[['identifier', 'domains']]
    rbps_df.columns = ['proteinID', 'domains']

    filt_rbps = annot_df['proteinID'].isin(rbps_df['proteinID'])
    annot_df.loc[filt_rbps, 'color'] = html_color
    annot_df.loc[filt_rbps, 'structural'] = 'RBP'
    annot_df.loc[filt_rbps, 'rgb_color_structural'] = rgb_color
    annot_df.loc[filt_rbps, 'RBP'] = True

    annot_df['RBP'].fillna(False, inplace=True)
    annot_df.loc[annot_df['RBP'], 'domains'] = rbps_df['domains'].to_list()
    annot_df['domains'].fillna('', inplace=True)
    return annot_df


def coGRR(animm_dir, phagedb_dir, output_dir):
    """ ... """

    proteins_dir = Path(phagedb_dir, 'split_records', 'proteins')
    process = run_ANImm(animm_dir, proteins_dir, output_dir)

    return process


def run_ANImm(animm_dir, input_dir, output_dir):
    """ ... """

    if Path(output_dir).exists():
        print(f'ANImm already done! To rerun delete folder: {output_dir}')
        return
    else: pass

    config_path = Path(animm_dir, 'sample-configs', 'proteins-based.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['input_dir'] = str(input_dir)
    config['output_dir'] = str(output_dir)

    with open(config_path, 'w+') as file:
        yaml.dump(config, file)

    print('Running ANImm... ', sep='')
    cmd = f'cd {animm_dir}; rm -rf .snakemake; nohup snakemake --use-conda --cores all --configfile sample-configs/proteins-based.yml &'
    process = run(cmd, capture_output=True, shell=True)
    print('Done!')
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

    cmd = f'bash -c "conda activate mashtree; mashtree.pl {fasta_dir}/* >> {output_file};"'

    print('Just run the command in bash. Problem with conda env AGAIN : / \n')
    print(cmd)

    process = run(cmd, capture_output=True, shell=True)

    return cmd


def better_mashtree(fasta_dir, tree, fnames=[]):
    """ generate newick mashtree and get leafs sorted by mashtree """

    Path(tree).parent.mkdir(exist_ok=True, parents=True)

    # preapre input files
    if fnames: files = [str(Path(fasta_dir, fname + '.fasta')) for fname in fnames]
    else: files = [str(path) for path in list(Path(fasta_dir).glob('*.fasta'))]

    input_files, tree = ' '.join(files), str(tree)

    cmd = f'bash -c "source activate root; conda activate mashtree; mashtree.pl {input_files} >> {tree};"'
    process = run(cmd, capture_output=True, shell=True)

    tree = list(Phylo.parse(tree, "newick"))[0]
    leafs = [leaf.name.upper() for leaf in tree.get_terminals()]

    return leafs


def tree2clades(mash_tree_path, phariants_table, clades, n_clusters=1, kmeans_show=True):
    """ ... """

    if Path(clades).exists():
        print(f'Mashtree already done! To rerun delete folder: {mash_tree_path}')
        return None
    else: pass

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


def table2genbank(annot_table, phages_fasta, genbank, phrogs_annot_table, rbps_table, colour_type='structural'):
    """ Converts output table (nr 5) from functional annotation to gebank file.
    phages_fasta is a path to file with all phage sequences.
    Decide which colour to put in genbank: 'structural' or 'phrogs'."""

    if genbank.exists():
        print(f'Nothing to do in table2genbank :)\nTo force rerun delete {genbank}')
        return None
    else: pass

    annot_df = merge_annot_table_and_phrogs_metatable(annot_table, phrogs_annot_table)
    annot_df = addRBPs(rbps_table, annot_df)
    print('Added RBPs predicted via domains to table :)')

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


def run_easyfig(genbank_dir, fnames, metadata_df, output_file, leg_name='structural',  columns2annotate=['phageID', 'backphlip', 'backphlip_conf', 'K_locus', 'ST', 'genetic_localisation', 'ICTV_Family'], script='other/Easyfig.py'):
    """ ... """

    # paths
    easyfig_dir, fstem = Path(output_file).parent, Path(output_file).stem
    genbank_dir = str( genbank_dir)

    raw_dir = Path(easyfig_dir, '1_raw')
    annotated_dir = Path(easyfig_dir, '2_annotated')

    # create output dirs
    raw_dir.mkdir(exist_ok=True, parents=True)
    annotated_dir.mkdir(exist_ok=True, parents=True)

    # split lists into pieces of 50 phages
    if len(fnames) > 30:
        pieces = len(fnames)/ 30
        fnames_pieces = list(map(list, np.array_split(fnames, pieces)))
    else:
        fnames_pieces = [fnames]

    # iterate over subparts of cluster(s)
    for piece, fnames in enumerate(fnames_pieces):

        svg = str(Path(raw_dir, fstem + f'_piece_{piece + 1}.svg'))
        jpeg_raw = Path(raw_dir, fstem + f'_piece_{piece + 1}.jpg')
        jpeg_annot = Path(annotated_dir, fstem + f'_piece_{piece + 1}.jpg')

        # prepare input files paths
        input_files = []
        for fname in fnames:
            path = str(Path(genbank_dir, fname.upper() + '.gb'))
            input_files.append(path)
        input_files = ' '.join(input_files)

        cmd = f'bash -c "source activate root; conda activate easyfig ; \
                python2 {script} \
                -svg -legend double -leg_name {leg_name} -f CDS -f1 T -i 60 -filter -aln right \
                -o {svg} {input_files};"'

        run(cmd, capture_output=True, shell=True)


        # convert svg2jpeg
        image = pyvips.Image.new_from_file(svg, dpi=100).flatten(background=255)
        image.write_to_file(jpeg_raw)

        # annotate figures
        text_color = (0,0,0)
        lines_to_annotate = get_text(metadata_df, fnames, columns2annotate)
        add_annotation(jpeg_raw, jpeg_annot, text_color, lines_to_annotate)


def between_phariants_easyfig(work_dir, clades, phages, phages_genbank_dir, leg_name='structural', columns2annotate=['phageID', 'backphlip', 'backphlip_conf', 'K_locus', 'ST', 'genetic_localisation', 'ICTV_Family']):
    """ ... """

    clades_df = pd.read_csv(clades, sep='\t')
    easyfig_dir = Path(work_dir, '1_coGRR', '3_easyfig', '1_between_phariants')
    clusters = sorted(list(clades_df['clusterID'].unique()))

    # checkpoint
    if easyfig_dir.exists():
        print('Nothing to do :)')
        print(f'To force rerun delete {str(easyfig_dir)}')
        return None, None
    else:
        print(f'Generating easyfig figures for {len(clusters)} clusters :) ', sep='')

    phages_df = pd.read_csv(phages, sep='\t')
    flip_phages = phages_df.loc[phages_df['flip_phage']]['phageID'].tolist()

    for nclust in clusters:
        leafs = clades_df.loc[clades_df['clusterID'] == nclust]['contigID'].to_list()
        phages_df = pd.read_csv(phages, sep='\t')
        phages_df.fillna('', inplace=True)

        svg = Path(easyfig_dir, f'clade_{str(nclust)}.svg')
        process = run_easyfig(phages_genbank_dir, leafs, phages_df, svg, leg_name='structural', columns2annotate=columns2annotate)

    print('Done! :)', sep='')
    return process


def get_text(df, phageIDs, columns):
    """ very nice """

    df.fillna('', inplace=True)
    df = df.loc[df['phageID'].isin(phageIDs)].copy()
    col_phageIDs = df['phageID'].to_list()
    lines = df.apply(lambda row: '  '.join(list(map(str, row[columns]))), axis=1).to_list()
    lines2annotate = {phageID: annotate for phageID, annotate in zip(col_phageIDs, lines)}
    lines2annotate = [lines2annotate[phageID] for phageID in phageIDs]
    return lines2annotate


def add_border(input_image, border, color):
    """ ... """
    img = Image.open(input_image)

    if isinstance(border, int) or isinstance(border, tuple):
        bimg = ImageOps.expand(img, border=border, fill=color)
    else:
        raise RuntimeError('Border is not an integer or tuple!')

    return bimg


def add_annotation(input_image, output_image, color, text_lines, font='other/arial.ttf'):
    """ Color as rgb tuple."""

    font = ImageFont.truetype(str(Path(font).resolve()), 70)

    add_pixels = 2500
    img = add_border(input_image, (0, 0, add_pixels, 0), color='white')
    draw = ImageDraw.Draw(img)
    xlocation, ylocation = img.size[0] - add_pixels + 100, - 10

    for text in text_lines:
        loc = (xlocation, ylocation)
        draw.text(loc, text, color, font)
        ylocation = ylocation + 208

    img.save(output_image)


def stGRR(animm_dir, phagedb_dir, output_dir, structural_categories=['head and packaging', 'connector', 'tail'], phrogs_table='other/upgraded_phrog_annot_v3.tsv'):
    """ ... """

    # if Path(output_dir).exists():
    #     print(f'stGRR already done! To rerun delete folder: {output_dir}')
    #     return
    # else: pass

    # paths
    annot_table = Path(phagedb_dir, 'annot.tsv')
    metadata_table = Path(phagedb_dir, 'phages.tsv')
    genbank_dir = Path(phagedb_dir, 'split_records/genbank')
    fasta_dir = Path(phagedb_dir, 'split_records/fasta')

    prots_dir = Path(output_dir, '1_struct_prots')
    stGRR_dir = Path(output_dir, '2_ANImm')
    clusters_dir = Path(output_dir, '3_clusters')

    struct_table = Path(output_dir, 'struct_annot.tsv')

    Path(prots_dir).mkdir(exist_ok=True, parents=True)
    Path(clusters_dir).mkdir(exist_ok=True, parents=True)

    # load tables
    annot_df = pd.read_csv(annot_table, sep='\t')
    phrogs_df = pd.read_csv(phrogs_table, sep='\t')

    annot_df['phrog'] = annot_df.apply(lambda row: int(row['target'].strip('phrog_')) , axis=1)
    annot_df = annot_df.merge(phrogs_df, on='phrog', how='left')

    filt_structural = annot_df['category'].isin(structural_categories)
    struct_df = annot_df.loc[filt_structural].groupby(['contigID', 'proteinID']).size().reset_index().iloc[:, :4]
    struct_df = struct_df.merge(annot_df, on=['contigID', 'proteinID'], how='left')
    struct_df.to_csv(struct_table, sep='\t', index=False)

    phages = annot_df['contigID'].unique()

    # save structural ORFs
    for phage in phages:
        filt_phage = (struct_df['contigID'] == phage)
        tmp_df = struct_df.loc[filt_phage]

        outfile = Path(prots_dir, phage + '.fasta')

        # if no structural proteins in phage save almost empty file
        if not len(tmp_df): continue
        else: pass

        proteinIDs, statuses, seqs = tmp_df['proteinID'], tmp_df['status'], tmp_df['protein']
        records = []

        for proteinID, status, seq in zip(proteinIDs, statuses, seqs):
            record = SeqRecord(Seq(seq), id=proteinID, description=status)
            records.append(record)

        SeqIO.write(records, outfile, 'fasta')
    print('Prepared input structural proteins :) ')

    # run ANImm on structural proteins
    proteins_dir = Path(prots_dir)

    if Path(stGRR_dir).exists():
        print(f'stGRR already done! To rerun ANImm delete folder: {stGRR_dir}')
        process = f'stGRR already done! To rerun ANImm delete folder: {stGRR_dir}'
    else:
        process = run_ANImm(animm_dir, proteins_dir, stGRR_dir)


    wgrr_thresholds = [0.9]
    print_tresholds = ', '.join(list(map(str, wgrr_thresholds)))
    print(f'Clustering by stGRR... (tresholds: {print_tresholds}) ', end='')
    stGRR_clusters(stGRR_dir, metadata_table, clusters_dir, wgrr_thresholds=wgrr_thresholds)
    print('Done!')

    print('Easyfig of stGRR clusters... ', end='')
    for threshold in wgrr_thresholds:
        easyfig_dir = Path(output_dir, '3_clusters', 'easyfig', f'wgrr-{threshold}')
        easyfig_dir.mkdir(exist_ok=True, parents=True)
        clusters_table = Path(clusters_dir, f'clusters-{threshold}.tsv')
        stGRR_easyfig(genbank_dir, fasta_dir, clusters_table, metadata_table, easyfig_dir)
    print('Done!')


def stGRR_clusters(stGRR_dir, metadata_table, clusters_dir, wgrr_thresholds):

    # get all phage IDs
    metadata_df = pd.read_csv(metadata_table, sep='\t')
    phageIDs = metadata_df['phageID']

    # wgrr
    wgrr_table = Path(stGRR_dir, 'wgrr.csv')
    wgrr_df = pd.read_csv(wgrr_table)

    output_paths = []
    for threshold in wgrr_thresholds:

        clusters_output = Path(clusters_dir, f'clusters-{threshold}.tsv')

        # filter table
        filt_wgrr = (wgrr_df['wgrr'] >= threshold)
        wgrr_tmp_df = wgrr_df.loc[filt_wgrr].loc[:, ['Seq1', 'Seq2']].copy()
        phage_nodes = list(set(wgrr_tmp_df['Seq1'].to_list() + wgrr_tmp_df['Seq2'].to_list()))

        # get missing singletons
        singletons = []
        for phageID in phageIDs:
            if phageID not in phage_nodes:
                singletons.append([phageID])

        # use networkx to generate the graph
        G = nx.Graph()
        G.add_edges_from(wgrr_tmp_df.to_numpy())

        # then get the adjacency matrix (in sparse form)
        matrix = nx.to_scipy_sparse_array(G)
        result = mc.run_mcl(matrix, inflation = 2)           # run MCL with default parameters
        clusters = mc.get_clusters(result)    # get clusters

        clusters.sort(key=lambda cluster: len(cluster), reverse=True)
        nodes = list(G.nodes)

        mapped_clusters = []
        for nodes_indicies in clusters:
            cluster_nodes = list(map(nodes.__getitem__, nodes_indicies))
            mapped_clusters.append(cluster_nodes)

        # add singletons
        mapped_clusters = mapped_clusters + singletons

        clusterIDs = []
        for i, cluster in enumerate(mapped_clusters):
            clusterIDs.append(len(cluster) * [i + 1])

        mapped_clusters = list(itertools.chain(*mapped_clusters))
        clusterIDs = list(itertools.chain(*clusterIDs))

        clusters_df = pd.DataFrame({'clusterID': clusterIDs, 'phageID': mapped_clusters})
        clusters_df.to_csv(clusters_output, sep='\t', index=False)

        output_paths.append(clusters_output)

    return output_paths


def stGRR_easyfig(genbank_dir, fasta_dir, clusters_table, metadata_table, easyfig_dir):

    clusters_df = pd.read_csv(clusters_table, sep='\t')
    metadata_df = pd.read_csv(metadata_table, sep='\t')

    trees_dir = Path(easyfig_dir, 'trees')
    trees_dir.mkdir(exist_ok=True, parents=True)

    clusters_grouped = clusters_df.groupby('clusterID').groups

    clusters_grouped_mapped = {}
    for clusterID in clusters_grouped.keys():
        clusters_grouped_mapped[clusterID] = clusters_df.loc[clusters_grouped[clusterID]]['phageID'].to_list()

    for clusterID in clusters_grouped_mapped.keys():
        outfile = Path(easyfig_dir, f'cluster_{clusterID}.svg')
        fnames = clusters_grouped_mapped[clusterID]

        if len(fnames) > 2:
            tree = Path(trees_dir, f'cluster_{clusterID}.newick')
            fnames = better_mashtree(fasta_dir, tree, fnames=fnames)
            run_easyfig(genbank_dir, fnames, metadata_df, outfile)


def GRR(coGRR_table, stGRR_table, metadata_table, genbank_dir, GRR_dir, network_grr_tresholds = [0.9, 0.8, 0.7, 0.6, 0.5], wgrr_co=0.4, wgrr_st=0.8, easyfig=False, show=False, force=False):

    if Path(GRR_dir).exists() and not force:
        print(f'GRR alread done! To rerun delete folder: {GRR_dir}')
        return None
    elif Path(GRR_dir).exists() or force:
        print(f'Running GRR!')
        pass
    else: print('Error!')

    print('Merging tables... ', end='')
    GRR_df = merge_coGRR_and_stGRR(coGRR_table, stGRR_table, GRR_dir)
    print('Done!')

    print('Generating networks... ', end='')
    network_dir = Path(GRR_dir, 'networks')
    network_dir.mkdir(exist_ok=True, parents=True)
    for grr_type in ['st', 'co']:
        for treshold in network_grr_tresholds:
            network = Path(network_dir, f'network-{grr_type}-{treshold}.sif')
            grr2network(GRR_df, network, grr_type, treshold)
    print('Done!')

    print('Generating correlation plot... ')
    meta_df = pd.read_csv(metadata_table, sep='\t')
    print('    1. prophages & inphared... ', end='')
    GRR_correlation(GRR_df, meta_df, GRR_dir, fname='corr-pp-inph.png')
    if not show: plt.close()
    print('Done!')

    print('    2. inphared... ', end='')
    filt_inphared = ~((GRR_df['Seq1'].str.contains('^[0-9]')) & (GRR_df['Seq2'].str.contains('^[0-9]')) & (GRR_df['Seq1'].str.contains('^[0-9]')) | (GRR_df['Seq2'].str.contains('^[0-9]')))
    GRR__inphared_df = GRR_df.loc[filt_inphared].copy()
    GRR_correlation(GRR__inphared_df, meta_df, GRR_dir, fname='corr-inph.png')
    if not show: plt.close()
    print('Done!')

    if easyfig:
        print('Easyfig... ', end='')
        easyfig_dir = Path(GRR_dir, 'easyfig')
        GRR_easyfig(GRR_df, metadata_table, genbank_dir, easyfig_dir, wgrr_co=wgrr_co, wgrr_st=wgrr_st)
        print('Done!')
    else: pass

    return GRR_df


def merge_coGRR_and_stGRR(coGRR_table, stGRR_table, GRR_dir):

    Path(GRR_dir).mkdir(exist_ok=True, parents=True)
    GRR_table = Path(GRR_dir, 'grr.csv')

    coGRR_df = pd.read_csv(coGRR_table)
    stGRR_df = pd.read_csv(stGRR_table)

    GRR_df = coGRR_df.merge(stGRR_df, on=['Seq1', 'Seq2'], suffixes=('_co', '_st'))
    GRR_df.to_csv(GRR_table, sep='\t', index=False)

    return GRR_df


def grr2network(GRR_df, network, structural_or_complete, grr_treshold):

    filt_grr = (GRR_df[f'wgrr_{structural_or_complete}'] >= grr_treshold)
    GRR_df = GRR_df.loc[filt_grr].copy()

    GRR_df['interaction'] = 'artificial'
    cols = ['Seq1', 'interaction', 'Seq2']

    GRR_df = GRR_df[cols]
    GRR_df.rename({'Seq1': 'phage_1', 'Seq2': 'phage_2'}, axis=1, inplace=True)
    GRR_df.to_csv(network, index=False, sep='\t', header=None)


def GRR_correlation(GRR_df, metadata_df, GRR_dir, fname='corr.png'):

    plot = Path(GRR_dir, fname)

    metadata_df = metadata_df[['phageID', 'backphlip']]
    metadata_df.rename({'phageID': 'Seq1', 'backphlip': 'backphlip1'}, axis=1, inplace=True)
    GRR_df = GRR_df.merge(metadata_df, on='Seq1', how='left')
    metadata_df.rename({'Seq1': 'Seq2', 'backphlip1': 'backphlip2'}, axis=1, inplace=True)
    GRR_df = GRR_df.merge(metadata_df, on='Seq2', how='left')

    filt_cross = ((GRR_df['backphlip1'] == 'virulent') & (GRR_df['backphlip2'] == 'temperate')) | ((GRR_df['backphlip1'] == 'temperate') & (GRR_df['backphlip2'] == 'virulent'))
    filt_virulent = ((GRR_df['backphlip1'] == 'virulent') & (GRR_df['backphlip2'] == 'virulent'))
    filt_temperate = ((GRR_df['backphlip1'] == 'temperate') & (GRR_df['backphlip2'] == 'temperate'))

    GRR_df.loc[filt_cross, 'phage_pair_status'] = 'cross'
    GRR_df.loc[filt_virulent, 'phage_pair_status'] = 'both virulent'
    GRR_df.loc[filt_temperate, 'phage_pair_status'] = 'both temperate'
    GRR_df['phage_pair_status'].fillna('one is none', inplace=True)

    fig, ax = plt.subplots(1,1, figsize=(40,40))

    colors = [sns.color_palette('pastel')[0], sns.color_palette('pastel')[3], sns.color_palette('pastel')[-3], sns.color_palette('pastel')[2]]
    cats = ['both temperate', 'both virulent', 'one is none', 'cross']
    colors = {cat: col for cat, col in zip(cats, colors)}

    graph = sns.scatterplot(data=GRR_df, x='wgrr_co', y='wgrr_st', hue='phage_pair_status', palette=colors, size=3, ax=ax)
    graph.axhline(1)
    graph.axvline(1)
    fig.savefig(plot)


def GRR_easyfig(GRR_df, metadata_table, genbank_dir, output_dir, wgrr_co, wgrr_st):

    filt = (GRR_df['wgrr_co'] <= wgrr_co) & (GRR_df['wgrr_st'] >= wgrr_st)
    subset_df = GRR_df.loc[filt][['Seq1', 'Seq2']].copy()

    pairs = list(zip(subset_df['Seq1'].to_list(), subset_df['Seq2'].to_list()))
    pairs = list(itertools.chain(*pairs))

    metadata_df = pd.read_csv(metadata_table, sep='\t')
    metadata_df.fillna('', inplace=True)
    flip_phages = metadata_df.loc[metadata_df['flip_phage']]['phageID']

    pieces = len(pairs)/20
    pairs_sliced = list(map(list, np.array_split(pairs, pieces)))
    for i, pairs in enumerate(pairs_sliced):
        outfile = Path(output_dir, f'{i+1}_coGRR_{wgrr_co}_stGRR_{wgrr_st}.svg')
        run_easyfig(genbank_dir, pairs, metadata_df, outfile, leg_name='structural',  columns2annotate=['phageID', 'backphlip', 'backphlip_conf', 'K_locus', 'ST', 'genetic_localisation', 'ICTV_Family'], script='other/Easyfig.py')
