import pandas as pd

def extract_snps_from_gwas_by_p(file_path, p_threshold=1e-5):
    valid_dat = []
    cols_name = None
    with open(file_path, 'r') as fp:
        for i, line in enumerate(fp):
            if i == 0:
                cols_name = line.strip().split()
                continue
            line = line.strip().split()
            p = line[-3]
            if p != 'NA' and float(p) < p_threshold:
                valid_dat.append(line)
    if len(valid_dat) == 0:
        return None
    res_df = pd.DataFrame(valid_dat, columns=cols_name)
    res_df = res_df.apply(pd.to_numeric, errors='ignore')
    return res_df
