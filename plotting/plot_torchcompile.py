import os
import re
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt

def parse_logs(log_dir):
    data = {}
    
    # Regex to extract model size and job ID from the filename
    file_pattern = re.compile(r'gipfel-throughput-(.+?)-50s-1n.*\.log')
    # Regex to extract iteration and throughput
    line_pattern = re.compile(r'iteration\s+(\d+)/.*throughput per GPU \(TFLOP/s/GPU\):\s+([\d\.]+)')
    
    for filename in os.listdir(log_dir):
        m_file = file_pattern.match(filename)
        if m_file:
            size_str = m_file.group(1)
            filepath = os.path.join(log_dir, filename)
            
            iters = []
            tflops = []
            
            with open(filepath, 'r') as f:
                for line in f:
                    m_line = line_pattern.search(line)
                    if m_line:
                        iters.append(int(m_line.group(1)))
                        tflops.append(float(m_line.group(2)))
            
            if iters:
                data[size_str] = (iters, tflops)
                
    return data

def sort_key(size_str):
    if size_str.endswith('m'):
        return float(size_str[:-1])
    elif size_str.endswith('b'):
        return float(size_str[:-1]) * 1000
    return float('inf')

def plot_data(baseline_dir, dirs, out_path):
    plt.figure(figsize=(10, 6))

    all_dirs = []
    if baseline_dir and os.path.exists(baseline_dir):
        all_dirs.append(baseline_dir)
    elif baseline_dir:
        print(f"Baseline directory {baseline_dir} does not exist.")

    for d in dirs:
        if os.path.exists(d):
            all_dirs.append(d)
        else:
            print(f"Directory {d} does not exist. Skipping.")

    all_data = {}
    all_sizes = set()
    for log_dir in all_dirs:
        data = parse_logs(log_dir)
        all_data[log_dir] = data
        all_sizes.update(data.keys())

    sorted_sizes = sorted(list(all_sizes), key=sort_key)
    try:
        cmap = mpl.colormaps['tab10']
    except AttributeError:
        cmap = plt.get_cmap('tab10')
    size_to_color = {size: cmap(i % 10) for i, size in enumerate(sorted_sizes)}

    markers = ['o', '^', 'x', 's', 'v', 'D', '<', '>']
    line_styles = ['-', '--', ':', '-.']

    for i, log_dir in enumerate(all_dirs):
        data = all_data[log_dir]
        dir_name = os.path.basename(os.path.normpath(log_dir))
        
        marker = markers[i % len(markers)]
        ls = line_styles[i % len(line_styles)]
        
        for size in sorted(data.keys(), key=sort_key):
            iters, tflops = data[size]
            label = f"{size} ({dir_name})" if len(all_dirs) > 1 else size
            plt.plot(iters, tflops, label=label, color=size_to_color[size], linestyle=ls, marker=marker, markersize=4)

    plt.xlabel("Iteration")
    plt.ylabel("Throughput per GPU (TFLOP/s/GPU)")
    plt.title("Throughput per Iteration")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot throughput per iteration from logs")
    parser.add_argument("--baseline", default="logs/baseline", help="Baseline directory containing log files")
    parser.add_argument("--dirs", nargs='+', default=["logs/torch.compile"], help="Directories containing log files")
    parser.add_argument("--output", default="plotting/plots/torchcompile.svg", help="Output plot filename")
    args = parser.parse_args()
    
    plot_data(args.baseline, args.dirs, args.output)
