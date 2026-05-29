import os
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np

def parse_logs(log_dir):
    data = {}
    
    # Regex to extract configuration from filename: gipfel-throughput-<size>-tp<tp>-pp<pp>-<n>n-<jobid>.log
    file_pattern = re.compile(r'gipfel-throughput-(.+?)-tp(\d+)-pp(\d+)-(\d+)n.*\.log')
    # Regex to extract iteration and throughput
    line_pattern = re.compile(r'iteration\s+(\d+)/.*throughput per GPU \(TFLOP/s/GPU\):\s+([\d\.]+).*tokens/sec/GPU:\s+([\d\.]+)')
    
    for filename in os.listdir(log_dir):
        if not filename.endswith('.log'):
            continue
            
        m_file = file_pattern.match(filename)
        if m_file:
            size_str = m_file.group(1)
            tp = int(m_file.group(2))
            pp = int(m_file.group(3))
            nodes = int(m_file.group(4))
            
            filepath = os.path.join(log_dir, filename)
            
            tflops = []
            tokens = []
            
            with open(filepath, 'r') as f:
                for line in f:
                    m_line = line_pattern.search(line)
                    if m_line:
                        # Skip the first few warmup iterations optionally, or take the mean of all
                        tflops.append(float(m_line.group(2)))
                        tokens.append(float(m_line.group(3)))
            
            if tflops and tokens:
                # We'll use the mean throughput (excluding first 5 iterations as warmup if possible)
                if len(tflops) > 5:
                    avg_tflop = np.mean(tflops[5:])
                    avg_token = np.mean(tokens[5:])
                else:
                    avg_tflop = np.mean(tflops)
                    avg_token = np.mean(tokens)
                    
                key = (size_str, tp, pp, nodes)
                data[key] = {'tflop': avg_tflop, 'token': avg_token}
                
    return data

def sort_size_key(size_str):
    if size_str.endswith('m'):
        return float(size_str[:-1])
    elif size_str.endswith('b'):
        return float(size_str[:-1]) * 1000
    return float('inf')

def plot_manynodes(data, out_dir):
    sizes = sorted(list(set(k[0] for k in data.keys())), key=sort_size_key)
    nodes_list = sorted(list(set(k[3] for k in data.keys())))
    
    if not sizes or not nodes_list:
        print("No data found to plot.")
        return

    x = np.arange(len(sizes))
    width = 0.35  # width of the bars
    
    # Plot TFLOP/s/GPU
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, nodes in enumerate(nodes_list):
        tflops = []
        for size in sizes:
            # Assuming TP=4, PP=4
            key = (size, 4, 4, nodes)
            tflops.append(data[key]['tflop'] if key in data else 0)
        
        offset = width * i
        rects = ax.bar(x + offset, tflops, width, label=f'{nodes} Nodes')
        
    ax.set_ylabel('Mean Throughput per GPU (TFLOP/s/GPU)')
    ax.set_title('Throughput by Model Size and Number of Nodes (TP=4, PP=4)')
    ax.set_xticks(x + width / 2 * (len(nodes_list) - 1))
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(axis='y')
    
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "manynodes_throughput_tflops.pdf"))
    plt.close()

    # Plot Tokens/sec/GPU
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, nodes in enumerate(nodes_list):
        tokens = []
        for size in sizes:
            # Assuming TP=4, PP=4
            key = (size, 4, 4, nodes)
            tokens.append(data[key]['token'] if key in data else 0)
            
        offset = width * i
        rects = ax.bar(x + offset, tokens, width, label=f'{nodes} Nodes')
        
    ax.set_ylabel('Mean Tokens per Second per GPU')
    ax.set_title('Tokens/sec by Model Size and Number of Nodes (TP=4, PP=4)')
    ax.set_xticks(x + width / 2 * (len(nodes_list) - 1))
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(axis='y')
    
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "manynodes_throughput_tokens.pdf"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot many nodes throughput logs")
    parser.add_argument('--log_dir', type=str, default='logs/manynodes', help='Directory containing log files')
    parser.add_argument('--out_dir', type=str, default='plotting/output_manynodes', help='Directory to save plots')
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    data = parse_logs(args.log_dir)
    print(f"Parsed {len(data)} log entries.")
    plot_manynodes(data, args.out_dir)
