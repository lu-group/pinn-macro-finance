> Under Construction

# Economics PINN: Deep learning for solving and estimating dynamic macro-finance models
The data and code for the paper [B. Fan, E. Qiao, A. Jiao, Z. Gu, W. Li, & L. Lu. Deep Learning for Solving and Estimating Dynamic Macro-Finance Models. *Computational Economics*, 65, 3885–3921, 2025.](https://doi.org/10.1007/s10614-024-10693-3)

## Code
- A Macroeconomic Model with a Financial Sector
  - [Forward PDE Problem](https://github.com/lu-group/pinn-macro-finance/blob/main/src/bs_forward.py)
  - [Inverse PDE Problem with One Unknown Variable](https://github.com/lu-group/pinn-macro-finance/blob/main/src/bs_inverse_1_var.py)
  - [Inverse PDE Problem with Two Unknown Variables](https://github.com/lu-group/pinn-macro-finance/blob/main/src/bs_inverse_2_var.py)
  - Reference solution for comparison can be found in the [hjb_data](https://github.com/lu-group/pinn-macro-finance/tree/main/data/hjb_data) folder
- A Model of Industrial Dynamics with Financial Frictions
  - [Forward PDE Problem](https://github.com/lu-group/pinn-macro-finance/blob/main/src/hjb_forward.py)
  - [Inverse PDE Problem with Two Unknown Variables](https://github.com/lu-group/pinn-macro-finance/blob/main/src/hjb_inverse_2_var.py)
  - [Inverse PDE Problem with Three Unknown Variables](https://github.com/lu-group/pinn-macro-finance/blob/main/src/hjb_inverse_3_var.py)
  - Reference solution for comparison can be found in the [bs_data](https://github.com/lu-group/pinn-macro-finance/tree/main/data/bs_data) folder

## Cite this work
If you use this data or code for academic research, you are encouraged to cite the following paper:
```
@article{fan2025deep,
  author  = {Fan, Benjamin and Qiao, Edward and Jiao, Anran and Gu, Zhouzhou and Li, Wenhao and Lu, Lu},
  title   = {Deep learning for solving and estimating dynamic macro-finance models},
  journal = {Computational Economics},
  volume  = {65},
  pages   = {3885–3921},
  year    = {2025},
  doi     = {https://doi.org/10.1007/s10614-024-10693-3}
}
```

## Questions
To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
