## Environment
Before running the Med-Transformer, please install the torch library.
[https://pytorch.org/get-started/locally/]

**************
## File description
- [`icd10_list.npy`](./icd10/icd10_list.npy): The list of ICD-10 codes trained in the provided Med-Transformer model. Each ICD-10 code's index + 1 is the corresponding numeric representation of the ICD-10 code.
- [`d_seqs.txt`](./example_data/d_seqs.txt): The ICD-10 codes' numeric representation of the patients in the example data, which is converted by the `icd10_list.npy` file.
- [`t_seqs.txt`](./example_data/t_seqs.txt): The dignosis time for each ICD-10 in the example data.

**************
Train and evaluate Med-Transformer
``` bash
python train.py
```

**************
Run Med-Transformer to extract the hidden information of the structured medical history records
``` bash
python clinical_history.py --disease_icds "I10," --d_seqs_path "./example_data/d_seqs.txt" --t_seqs_path "./example_data/t_seqs.txt" --save_dir "./results"
```