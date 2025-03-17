import pandas as pd
import pybedtools
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to data
chia_pet_file = "data/4DNFIS9CCN6R_CHIAPET_LOOPS.bedpe/4DNFIS9CCN6R_CHIAPET_LOOPS.bedpe"
ctcf_chip_file = "data/ENCFF356LIU_CTCF_CHIPSEQ.bed/ENCFF356LIU_CTCF_CHIPSEQ.bed"
rad21_chip_file = "data/ENCFF834GOT_RAD21_CHIPSEQ.bed/ENCFF834GOT_RAD21_CHIPSEQ.bed"

# Load ChIA-PET loops (BEDPE format)
def load_bedpe(file_path):
    cols = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
    df = pd.read_csv(file_path, sep="\t", header=None, usecols=[0, 1, 2, 3, 4, 5], names=cols)
    return df

# Load ChIP-seq peaks (BED format)
def load_bed(file_path):
    cols = ["chrom", "start", "end"]
    df = pd.read_csv(file_path, sep="\t", header=None, usecols=[0, 1, 2], names=cols)
    return df

# Convert pandas DataFrame to BedTool for intersection
def df_to_bedtool(df):
    return pybedtools.BedTool.from_dataframe(df)

# Find loops with at least one or both anchors overlapping ChIP-seq peaks
def filter_loops(chia_pet, chip_seq, protein_name):
    chip_tool = df_to_bedtool(chip_seq)

    # Create BedTool objects for loop anchors
    anchor1 = df_to_bedtool(chia_pet[["chrom1", "start1", "end1"]])
    anchor2 = df_to_bedtool(chia_pet[["chrom2", "start2", "end2"]])

    # Find overlaps
    overlap1 = anchor1.intersect(chip_tool, wa=True).to_dataframe(names=["chrom1", "start1", "end1"])
    overlap2 = anchor2.intersect(chip_tool, wa=True).to_dataframe(names=["chrom2", "start2", "end2"])

    # Merge results
    chia_pet["{}_one_anchor".format(protein_name)] = chia_pet.apply(
        lambda row: any((row.chrom1 == overlap1.chrom1) & (row.start1 == overlap1.start1)) or
                    any((row.chrom2 == overlap2.chrom2) & (row.start2 == overlap2.start2)),
        axis=1
    )
    chia_pet["{}_both_anchors".format(protein_name)] = chia_pet.apply(
        lambda row: any((row.chrom1 == overlap1.chrom1) & (row.start1 == overlap1.start1)) and
                    any((row.chrom2 == overlap2.chrom2) & (row.start2 == overlap2.start2)),
        axis=1
    )

    return chia_pet

# Load data
chia_pet = load_bedpe(chia_pet_file)
ctcf_chip = load_bed(ctcf_chip_file)
rad21_chip = load_bed(rad21_chip_file)

# Filter loops
chia_pet = filter_loops(chia_pet, ctcf_chip, "CTCF")
chia_pet = filter_loops(chia_pet, rad21_chip, "Rad21")

# Identify common loops
common_loops = chia_pet[
    (chia_pet["CTCF_one_anchor"] & chia_pet["Rad21_one_anchor"]) |
    (chia_pet["CTCF_both_anchors"] & chia_pet["Rad21_both_anchors"])
]


# Summary statistics
print(f"Total loops: {len(chia_pet)}")
print(f"Loops with CTCF in one anchor: {chia_pet['CTCF_one_anchor'].sum()}")
print(f"Loops with CTCF in both anchors: {chia_pet['CTCF_both_anchors'].sum()}")
print(f"Loops with Rad21 in one anchor: {chia_pet['Rad21_one_anchor'].sum()}")
print(f"Loops with Rad21 in both anchors: {chia_pet['Rad21_both_anchors'].sum()}")
print(f"Common loops (CTCF & Rad21): {len(common_loops)}")

# Visualization
plt.figure(figsize=(8, 5))
sns.barplot(x=["CTCF One", "CTCF Both", "Rad21 One", "Rad21 Both", "Common"], 
            y=[chia_pet["CTCF_one_anchor"].sum(), chia_pet["CTCF_both_anchors"].sum(),
               chia_pet["Rad21_one_anchor"].sum(), chia_pet["Rad21_both_anchors"].sum(),
               len(common_loops)])
plt.ylabel("Number of Loops")
plt.title("Chromatin Loop Filtering Results")
plt.show()
