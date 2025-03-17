import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pybedtools
from matplotlib.patches import Arc
from matplotlib.collections import PatchCollection
import random
import gzip
import shutil
from upsetplot import UpSet

def download_data(download_dir="data"):
    """
    Download CTCF and Rad21 ChIP-seq data if not already present.
    Also download ChIA-PET loops data if not present.
    """
    import requests
    import os
    
    os.makedirs(download_dir, exist_ok=True)
    
    files_to_download = {
        "CTCF ChIP-seq": {
            "url": "https://www.encodeproject.org/files/ENCFF356LIU/@@download/ENCFF356LIU.bed.gz",
            "local_path": os.path.join(download_dir, "ENCFF356LIU_CTCF_CHIPSEQ.bed.gz")
        },
        "Rad21 ChIP-seq": {
            "url": "https://www.encodeproject.org/files/ENCFF834GOT/@@download/ENCFF834GOT.bed.gz",
            "local_path": os.path.join(download_dir, "ENCFF834GOT_RAD21_CHIPSEQ.bed.gz")
        },
        "ChIA-PET Loops": {
            "url": "https://data.4dnucleome.org/files-processed/4DNFIS9CCN6R/@@download/4DNFIS9CCN6R.bedpe.gz",
            "local_path": os.path.join(download_dir, "4DNFIS9CCN6R_CHIAPET_LOOPS.bedpe.gz")
        }
    }
    
    for name, file_info in files_to_download.items():
        uncompressed_path = file_info["local_path"].replace(".gz", "")
        # If uncompressed file exists, skip
        if os.path.exists(uncompressed_path):
            print(f"{name} file already exists at {uncompressed_path}")
            continue
        
        # If compressed file doesn't exist, download it
        if not os.path.exists(file_info["local_path"]):
            print(f"Downloading {name} from {file_info['url']}...")
            r = requests.get(file_info["url"], stream=True)
            with open(file_info["local_path"], 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded {name} to {file_info['local_path']}")
        else:
            print(f"{name} compressed file already exists at {file_info['local_path']}")
        
        # Uncompress the file
        print(f"Decompressing {file_info['local_path']}...")
        with gzip.open(file_info["local_path"], 'rb') as f_in:
            with open(uncompressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Uncompressed {name} to {uncompressed_path}")
    
    return {
        "chia_pet_file": os.path.join(download_dir, "4DNFIS9CCN6R_CHIAPET_LOOPS.bedpe"),
        "ctcf_chip_file": os.path.join(download_dir, "ENCFF356LIU_CTCF_CHIPSEQ.bed"),
        "rad21_chip_file": os.path.join(download_dir, "ENCFF834GOT_RAD21_CHIPSEQ.bed")
    }

def load_bedpe(file_path):
    """Load ChIA-PET loops (BEDPE format)"""
    cols = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
    df = pd.read_csv(file_path, sep="\t", header=None, usecols=[0, 1, 2, 3, 4, 5], names=cols)
    # Add unique ID and loop length
    df['loop_id'] = [f"loop_{i}" for i in range(len(df))]
    df['loop_length'] = df['start2'] - df['end1']
    return df

def load_bed(file_path):
    """Load ChIP-seq peaks (BED format)"""
    cols = ["chrom", "start", "end"]
    df = pd.read_csv(file_path, sep="\t", header=None, usecols=[0, 1, 2], names=cols)
    return df

def filter_loops(chia_pet, chip_seq, protein_name):
    """
    Find loops with overlapping anchors efficiently
    Returns the original dataframe with added columns and filtered dataframes
    """
    # Create temporary files for BedTools operations
    chip_tool = pybedtools.BedTool.from_dataframe(chip_seq)
    
    # Prepare anchor dataframes with index for tracking
    anchor1 = chia_pet[["chrom1", "start1", "end1", "loop_id"]].copy()
    anchor2 = chia_pet[["chrom2", "start2", "end2", "loop_id"]].copy()
    
    # Convert to BedTool
    anchor1_bed = pybedtools.BedTool.from_dataframe(anchor1)
    anchor2_bed = pybedtools.BedTool.from_dataframe(anchor2)
    
    # Find overlaps and convert back to dataframes
    overlap1 = anchor1_bed.intersect(chip_tool, u=True).to_dataframe()
    overlap2 = anchor2_bed.intersect(chip_tool, u=True).to_dataframe()
    
    if len(overlap1.columns) > 0 and len(overlap1) > 0:
        overlap1.columns = ["chrom1", "start1", "end1", "loop_id"]
    if len(overlap2.columns) > 0 and len(overlap2) > 0:
        overlap2.columns = ["chrom2", "start2", "end2", "loop_id"]
    
    # Create sets of loop_ids for efficient lookup
    ids1_set = set(overlap1['loop_id']) if 'loop_id' in overlap1.columns and len(overlap1) > 0 else set()
    ids2_set = set(overlap2['loop_id']) if 'loop_id' in overlap2.columns and len(overlap2) > 0 else set()
    
    # Create boolean arrays using sets
    one_anchor = np.array([loop_id in ids1_set or loop_id in ids2_set for loop_id in chia_pet['loop_id']])
    both_anchors = np.array([loop_id in ids1_set and loop_id in ids2_set for loop_id in chia_pet['loop_id']])
    
    # Add columns to result dataframe
    chia_pet[f"{protein_name}_one_anchor"] = one_anchor
    chia_pet[f"{protein_name}_both_anchors"] = both_anchors
    
    # Track which specific anchor overlaps with the chip-seq peaks
    chia_pet[f"{protein_name}_anchor1"] = np.array([loop_id in ids1_set for loop_id in chia_pet['loop_id']])
    chia_pet[f"{protein_name}_anchor2"] = np.array([loop_id in ids2_set for loop_id in chia_pet['loop_id']])
    
    # Create filtered datasets
    loops_one_anchor = chia_pet[chia_pet[f"{protein_name}_one_anchor"]].copy()
    loops_both_anchors = chia_pet[chia_pet[f"{protein_name}_both_anchors"]].copy()
    
    return chia_pet, loops_one_anchor, loops_both_anchors

def analyze_loop_overlaps(chia_pet, output_dir="results"):
    """Generate comprehensive statistics and save them to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic counts
    total_loops = len(chia_pet)
    ctcf_one = chia_pet["CTCF_one_anchor"].sum()
    ctcf_both = chia_pet["CTCF_both_anchors"].sum()
    rad21_one = chia_pet["Rad21_one_anchor"].sum()
    rad21_both = chia_pet["Rad21_both_anchors"].sum()
    
    # Common loops
    ctcf_rad21_one = ((chia_pet["CTCF_one_anchor"]) & (chia_pet["Rad21_one_anchor"])).sum()
    ctcf_rad21_both = ((chia_pet["CTCF_both_anchors"]) & (chia_pet["Rad21_both_anchors"])).sum()
    
    # Specific anchor overlaps
    both_ctcf_anchor1 = chia_pet["CTCF_anchor1"].sum()
    both_ctcf_anchor2 = chia_pet["CTCF_anchor2"].sum()
    both_rad21_anchor1 = chia_pet["Rad21_anchor1"].sum()
    both_rad21_anchor2 = chia_pet["Rad21_anchor2"].sum()
    
    # Shared anchor patterns
    both_proteins_anchor1 = ((chia_pet["CTCF_anchor1"]) & (chia_pet["Rad21_anchor1"])).sum()
    both_proteins_anchor2 = ((chia_pet["CTCF_anchor2"]) & (chia_pet["Rad21_anchor2"])).sum()
    ctcf_anchor1_rad21_anchor2 = ((chia_pet["CTCF_anchor1"]) & (chia_pet["Rad21_anchor2"])).sum()
    ctcf_anchor2_rad21_anchor1 = ((chia_pet["CTCF_anchor2"]) & (chia_pet["Rad21_anchor1"])).sum()
    
    # Prepare statistics table
    stats = pd.DataFrame({
        'Metric': [
            'Total loops',
            'Loops with CTCF in at least one anchor',
            'Loops with CTCF in both anchors',
            'Loops with Rad21 in at least one anchor',
            'Loops with Rad21 in both anchors',
            'Loops with both CTCF and Rad21 in at least one anchor',
            'Loops with both CTCF and Rad21 in both anchors',
            'Loops with CTCF in anchor 1',
            'Loops with CTCF in anchor 2',
            'Loops with Rad21 in anchor 1',
            'Loops with Rad21 in anchor 2',
            'Loops with both CTCF and Rad21 in anchor 1',
            'Loops with both CTCF and Rad21 in anchor 2',
            'Loops with CTCF in anchor 1 and Rad21 in anchor 2',
            'Loops with CTCF in anchor 2 and Rad21 in anchor 1'
        ],
        'Count': [
            total_loops,
            ctcf_one,
            ctcf_both,
            rad21_one,
            rad21_both,
            ctcf_rad21_one,
            ctcf_rad21_both,
            both_ctcf_anchor1,
            both_ctcf_anchor2,
            both_rad21_anchor1,
            both_rad21_anchor2,
            both_proteins_anchor1,
            both_proteins_anchor2,
            ctcf_anchor1_rad21_anchor2,
            ctcf_anchor2_rad21_anchor1
        ],
        'Percentage': [
            100.0,
            ctcf_one / total_loops * 100,
            ctcf_both / total_loops * 100,
            rad21_one / total_loops * 100,
            rad21_both / total_loops * 100,
            ctcf_rad21_one / total_loops * 100,
            ctcf_rad21_both / total_loops * 100,
            both_ctcf_anchor1 / total_loops * 100,
            both_ctcf_anchor2 / total_loops * 100,
            both_rad21_anchor1 / total_loops * 100,
            both_rad21_anchor2 / total_loops * 100,
            both_proteins_anchor1 / total_loops * 100,
            both_proteins_anchor2 / total_loops * 100,
            ctcf_anchor1_rad21_anchor2 / total_loops * 100,
            ctcf_anchor2_rad21_anchor1 / total_loops * 100
        ]
    })
    
    # Save statistics
    stats.to_csv(os.path.join(output_dir, "loop_statistics.csv"), index=False)
    
    # Create summary text file
    with open(os.path.join(output_dir, "summary_statistics.txt"), "w") as f:
        f.write("CHROMATIN LOOP ANALYSIS SUMMARY\n")
        f.write("==============================\n\n")
        f.write(f"Total ChIA-PET loops analyzed: {total_loops}\n\n")
        
        f.write("CTCF BINDING:\n")
        f.write(f"- Loops with CTCF in at least one anchor: {ctcf_one} ({ctcf_one/total_loops:.2%})\n")
        f.write(f"- Loops with CTCF in both anchors: {ctcf_both} ({ctcf_both/total_loops:.2%})\n")
        f.write(f"- Loops with CTCF in anchor 1: {both_ctcf_anchor1} ({both_ctcf_anchor1/total_loops:.2%})\n")
        f.write(f"- Loops with CTCF in anchor 2: {both_ctcf_anchor2} ({both_ctcf_anchor2/total_loops:.2%})\n\n")
        
        f.write("Rad21 BINDING:\n")
        f.write(f"- Loops with Rad21 in at least one anchor: {rad21_one} ({rad21_one/total_loops:.2%})\n")
        f.write(f"- Loops with Rad21 in both anchors: {rad21_both} ({rad21_both/total_loops:.2%})\n")
        f.write(f"- Loops with Rad21 in anchor 1: {both_rad21_anchor1} ({both_rad21_anchor1/total_loops:.2%})\n")
        f.write(f"- Loops with Rad21 in anchor 2: {both_rad21_anchor2} ({both_rad21_anchor2/total_loops:.2%})\n\n")
        
        f.write("CO-BINDING PATTERNS:\n")
        f.write(f"- Loops with both CTCF and Rad21 in at least one anchor: {ctcf_rad21_one} ({ctcf_rad21_one/total_loops:.2%})\n")
        f.write(f"- Loops with both CTCF and Rad21 in both anchors: {ctcf_rad21_both} ({ctcf_rad21_both/total_loops:.2%})\n")
        f.write(f"- Loops with both CTCF and Rad21 in anchor 1: {both_proteins_anchor1} ({both_proteins_anchor1/total_loops:.2%})\n")
        f.write(f"- Loops with both CTCF and Rad21 in anchor 2: {both_proteins_anchor2} ({both_proteins_anchor2/total_loops:.2%})\n")
        f.write(f"- Loops with CTCF in anchor 1 and Rad21 in anchor 2: {ctcf_anchor1_rad21_anchor2} ({ctcf_anchor1_rad21_anchor2/total_loops:.2%})\n")
        f.write(f"- Loops with CTCF in anchor 2 and Rad21 in anchor 1: {ctcf_anchor2_rad21_anchor1} ({ctcf_anchor2_rad21_anchor1/total_loops:.2%})\n")
    
    return stats

def visualize_loop_statistics(chia_pet, output_dir="results"):
    """Create various visualizations of the loop statistics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Bar plot comparing loop counts
    plt.figure(figsize=(12, 6))
    counts = [
        chia_pet["CTCF_one_anchor"].sum(),
        chia_pet["CTCF_both_anchors"].sum(),
        chia_pet["Rad21_one_anchor"].sum(),
        chia_pet["Rad21_both_anchors"].sum(),
        ((chia_pet["CTCF_one_anchor"]) & (chia_pet["Rad21_one_anchor"])).sum(),
        ((chia_pet["CTCF_both_anchors"]) & (chia_pet["Rad21_both_anchors"])).sum()
    ]
    
    labels = [
        "CTCF\n≥1 Anchor", 
        "CTCF\nBoth Anchors",
        "Rad21\n≥1 Anchor", 
        "Rad21\nBoth Anchors",
        "CTCF+Rad21\n≥1 Anchor",
        "CTCF+Rad21\nBoth Anchors"
    ]
    
    # Add percentage labels
    total = len(chia_pet)
    percentages = [count / total * 100 for count in counts]
    
    ax = sns.barplot(x=labels, y=counts)
    plt.ylabel("Number of Loops")
    plt.title("Chromatin Loop Binding Patterns")
    plt.xticks(rotation=45, ha="right")  # Rotated labels for better visibility
    
    # Add count and percentage labels to bars
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2.,
                height + 0.1,
                f'{counts[i]}\n({percentages[i]:.1f}%)',
                ha="center", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loop_count_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Upset plot (alternative to Venn diagram for multiple sets)
    try:
        # Create binary indicators for each category
        data = pd.DataFrame({
            'CTCF Anchor 1': chia_pet["CTCF_one_anchor"],
            'CTCF Anchor 2': chia_pet["CTCF_both_anchors"],
            'Rad21 Anchor 1': chia_pet["Rad21_one_anchor"],
            'Rad21 Anchor 2': chia_pet["Rad21_both_anchors"]
        })
        
        # Count combinations
        combinations = data.groupby(list(data.columns)).size()
        
        plt.figure(figsize=(12, 6))
        upset = UpSet(combinations, 
                      sort_by='cardinality',
                      show_percentages=True
                      )
        upset.plot()
        plt.title("Intersections of Protein Binding at Loop Anchors")
        
        # Rotate annotations
        plt.xticks(rotation=45, ha="right")  # Ensuring visibility
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "upset_plot.png"), dpi=300, bbox_inches="tight")
        plt.close()
    except ImportError:
        print("upsetplot package not installed. Skipping upset plot.")
    
    # 3. Loop length distribution by binding pattern
    plt.figure(figsize=(10, 6))
    
    # Define categories for comparison
    categories = {
        "All loops": chia_pet,
        "CTCF both anchors": chia_pet[chia_pet["CTCF_both_anchors"]],
        "Rad21 both anchors": chia_pet[chia_pet["Rad21_both_anchors"]],
        "CTCF+Rad21 both": chia_pet[(chia_pet["CTCF_both_anchors"]) & (chia_pet["Rad21_both_anchors"])]
    }
    
    # Plot loop length distributions
    for name, df in categories.items():
        if len(df) > 0:  # Only plot if there are loops in this category
            sns.kdeplot(df['loop_length'], label=f"{name} (n={len(df)})")
    
    plt.xlabel("Loop Length (bp)")
    plt.ylabel("Density")
    plt.title("Distribution of Loop Lengths by Binding Pattern")
    plt.legend(fontsize="small")  # Making legend readable
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loop_length_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    return True

def visualize_genomic_region(loops_df, ctcf_df, rad21_df, region=None, output_dir="results"):
    """
    Visualize chromatin loops in a specific genomic region
    
    Parameters:
        loops_df: DataFrame with loop data
        ctcf_df: DataFrame with CTCF ChIP-seq peaks
        rad21_df: DataFrame with Rad21 ChIP-seq peaks
        region: dict with chromosome, start, end (if None, will select a region with loops)
        output_dir: directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # If no region specified, find one with loops
    if region is None:
        # Get a random chromosome with loops
        chr_choices = loops_df["chrom1"].unique()
        if len(chr_choices) == 0:
            print("No loops found to visualize")
            return False
        
        chosen_chr = random.choice(chr_choices)
        
        # Find loops on this chromosome
        chr_loops = loops_df[loops_df["chrom1"] == chosen_chr]
        
        if len(chr_loops) == 0:
            print(f"No loops found on chromosome {chosen_chr}")
            return False
            
        # Choose a random starting point near some loops
        random_loop = chr_loops.sample(1).iloc[0]
        region_mid = (random_loop["start1"] + random_loop["end2"]) // 2
        region_size = max(2000000, random_loop["end2"] - random_loop["start1"] + 1000000)
        
        region = {
            "chrom": chosen_chr,
            "start": max(0, region_mid - region_size//2),
            "end": region_mid + region_size//2
        }
    
    # Filter data for the selected region
    region_loops = loops_df[
        (loops_df["chrom1"] == region["chrom"]) & 
        (loops_df["start1"] >= region["start"]) & 
        (loops_df["end2"] <= region["end"])
    ]
    
    region_ctcf = ctcf_df[
        (ctcf_df["chrom"] == region["chrom"]) & 
        (ctcf_df["start"] >= region["start"]) & 
        (ctcf_df["end"] <= region["end"])
    ]
    
    region_rad21 = rad21_df[
        (rad21_df["chrom"] == region["chrom"]) & 
        (rad21_df["start"] >= region["start"]) & 
        (rad21_df["end"] <= region["end"])
    ]
    
    if len(region_loops) == 0:
        print(f"No loops found in region {region['chrom']}:{region['start']}-{region['end']}")
        return False
    
    print(f"Visualizing {len(region_loops)} loops in {region['chrom']}:{region['start']}-{region['end']}")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Normalize positions to 0-1 range
    region_width = region["end"] - region["start"]
    
    # Plot loops as arcs
    arcs = []
    for _, loop in region_loops.iterrows():
        # Calculate loop midpoint and width
        start = (loop["start1"] - region["start"]) / region_width
        end = (loop["end2"] - region["start"]) / region_width
        
        # Create arc
        height = min(0.4, (end - start) * 2)  # Scale height based on width
        arc = Arc((start + (end - start)/2, 0), 
                 width=end-start, 
                 height=height, 
                 theta1=0, 
                 theta2=180, 
                 linewidth=1.5)
        
        # Color based on binding pattern
        if loop.get("CTCF_both_anchors", False) and loop.get("Rad21_both_anchors", False):
            arc.set_color('purple')  # Both proteins on both anchors
        elif loop.get("CTCF_both_anchors", False):
            arc.set_color('blue')    # CTCF on both anchors
        elif loop.get("Rad21_both_anchors", False):
            arc.set_color('green')   # Rad21 on both anchors
        elif loop.get("CTCF_one_anchor", False) and loop.get("Rad21_one_anchor", False):
            arc.set_color('orange')  # Both proteins on at least one anchor
        elif loop.get("CTCF_one_anchor", False):
            arc.set_color('lightblue')  # CTCF on one anchor
        elif loop.get("Rad21_one_anchor", False):
            arc.set_color('lightgreen')  # Rad21 on one anchor
        else:
            arc.set_color('grey')    # No protein binding
        
        arcs.append(arc)
    
    # Add all arcs to plot
    for arc in arcs:
        ax.add_patch(arc)
    
    # Plot CTCF and Rad21 peaks
    peak_height = -0.02
    
    # Plot CTCF peaks
    for _, peak in region_ctcf.iterrows():
        start = (peak["start"] - region["start"]) / region_width
        end = (peak["end"] - region["start"]) / region_width
        ax.add_patch(plt.Rectangle((start, peak_height), end-start, 0.02, color='blue', alpha=0.7))
    
    # Plot Rad21 peaks
    peak_height -= 0.03
    for _, peak in region_rad21.iterrows():
        start = (peak["start"] - region["start"]) / region_width
        end = (peak["end"] - region["start"]) / region_width
        ax.add_patch(plt.Rectangle((start, peak_height), end-start, 0.02, color='green', alpha=0.7))
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], color='blue', label='CTCF both anchors'),
        plt.Line2D([0], [0], color='green', label='Rad21 both anchors'),
        plt.Line2D([0], [0], color='purple', label='CTCF+Rad21 both anchors'),
        plt.Line2D([0], [0], color='lightblue', label='CTCF one anchor'),
        plt.Line2D([0], [0], color='lightgreen', label='Rad21 one anchor'),
        plt.Line2D([0], [0], color='orange', label='CTCF+Rad21 one anchor'),
        plt.Line2D([0], [0], color='grey', label='No protein binding'),
        plt.Rectangle((0, 0), 1, 1, color='blue', alpha=0.7, label='CTCF ChIP-seq peak'),
        plt.Rectangle((0, 0), 1, 1, color='green', alpha=0.7, label='Rad21 ChIP-seq peak')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    # Set axis limits and labels
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 0.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels([
        f"{region['start']:,}",
        f"{region['start'] + region_width//4:,}",
        f"{region['start'] + region_width//2:,}",
        f"{region['start'] + 3*region_width//4:,}",
        f"{region['end']:,}"
    ])
    ax.set_yticks([])
    
    # Add title and labels
    plt.title(f"Chromatin Loops and Protein Binding at {region['chrom']}:{region['start']:,}-{region['end']:,}")
    plt.xlabel(f"Position on {region['chrom']} (bp)")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"genomic_region_{region['chrom']}_{region['start']}-{region['end']}.png"), dpi=300)
    plt.close()
    
    # Save region data to BED format for potential visualization in genome browsers
    with open(os.path.join(output_dir, "example_region.bed"), "w") as f:
        f.write(f"{region['chrom']}\t{region['start']}\t{region['end']}\tExample_Region\n")
    
    # Also save a summary text file with the region information
    with open(os.path.join(output_dir, "example_region_info.txt"), "w") as f:
        f.write(f"Region: {region['chrom']}:{region['start']}-{region['end']}\n")
        f.write(f"Total loops in region: {len(region_loops)}\n")
        
        # Count different types of loops
        ctcf_one = region_loops["CTCF_one_anchor"].sum() if "CTCF_one_anchor" in region_loops.columns else 0
        ctcf_both = region_loops["CTCF_both_anchors"].sum() if "CTCF_both_anchors" in region_loops.columns else 0
        rad21_one = region_loops["Rad21_one_anchor"].sum() if "Rad21_one_anchor" in region_loops.columns else 0
        rad21_both = region_loops["Rad21_both_anchors"].sum() if "Rad21_both_anchors" in region_loops.columns else 0

        f.write(f"Loops with CTCF in at least one anchor: {ctcf_one}\n")
        f.write(f"Loops with CTCF in both anchors: {ctcf_both}\n")
        f.write(f"Loops with Rad21 in at least one anchor: {rad21_one}\n")
        f.write(f"Loops with Rad21 in both anchors: {rad21_both}\n")

